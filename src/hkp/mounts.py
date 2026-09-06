"""Public service endpoints, served by the shared runtime server.

Mirrors hkp-node's ``src/mounts.ts``.
"""
from __future__ import annotations

import hmac
import secrets
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Awaitable, Callable, Protocol

from aiohttp import web

#: Requests under this prefix are served by mounts rather than the REST API.
MOUNT_PREFIX = "/hosted"


@dataclass
class MountContext:
    """Where a mount is served and how much of the request path belongs to it."""

    #: Public path prefix this mount owns, e.g. ``/hosted/ab12…``.
    mount_path: str
    #: Request target with ``mount_path`` removed; always starts with ``/`` and
    #: keeps any query string, so a handler can parse it as a URL and see both.
    sub_path: str


MountHandler = Callable[[web.Request, MountContext], Awaitable[web.StreamResponse]]


@dataclass
class MountHandle:
    """A live mount, handed back to the service that registered it."""

    #: Public URL clients should be pointed at.
    url: str
    #: Path prefix of ``url``, for clients configured by host/port/path.
    path: str
    release: Callable[[], None]


@dataclass
class _MountRecord:
    owner: str
    runtime_id: str
    service_uuid: str
    handler: MountHandler


class MountRegistry:
    """Routes public traffic to services that need to expose an endpoint,
    without any of them binding a port of their own.

    A service asks for a mount and gets back an opaque, server-assigned id. That
    id — rather than a port or a caller-chosen path — is what makes the endpoint
    addressable, which matters for three reasons:

    - Ports are a single machine-wide namespace. With several tenants on one
      host, a service asking for a specific port is a land grab: the second
      claimant fails, and whoever wins receives traffic the other expected.
    - Runtime ids are only unique per tenant (boards ship stable ones like
      ``node``), so they cannot appear in a globally-routable path.
    - These endpoints are deliberately unauthenticated — they exist to be called
      by outside parties. An unguessable id therefore doubles as the capability
      to reach them, and it carries no user identifier that a public URL would
      otherwise leak.

    The id is **derived, not drawn**: an HMAC of who owns the mount, which board
    and runtime it is in, and what it is called, keyed by a secret only the
    server holds. Randomness would be just as unguessable and was what this did
    first, but it made the address change every time a board was loaded — so
    anything outside pointing at it (a webhook configured in somebody else's
    product) broke on every restart, and a board could not be redeployed without
    reconfiguring its callers.

    Deriving it keeps the address stable across reloads, restarts and redeploys
    while keeping the secret out of the board: the board says only what the mount
    is *called*, which is not sensitive, and the server turns that into an
    address nobody can compute without the key. Rotating the key rotates every
    endpoint, and renaming one mount rotates only that one.
    """

    def __init__(
        self,
        public_url_for: Callable[[str], str | None],
        secret: str | None = None,
    ) -> None:
        #: Resolves a mount path to the URL clients should use. Returns None
        #: before the server is listening, since the port is not known until then.
        self._public_url_for = public_url_for
        #: Keys the id derivation. A server that is given none draws one for this
        #: process, which is the old behaviour: addresses that work but do not
        #: survive a restart. ``__main__`` persists one so they do.
        self._secret = secret or secrets.token_hex(32)
        self._mounts: dict[str, _MountRecord] = {}

    def _derive_id(
        self,
        owner: str,
        board_name: str,
        runtime_id: str,
        name: str,
    ) -> str:
        """The address a given mount always gets.

        Every part that identifies the mount goes in, so two services cannot
        derive the same id: the tenant, the board, the runtime, and the mount's
        name. NUL separates them, as it cannot occur in any of them — the same
        reasoning as ``_tenant_key`` in the server.
        """
        message = "\u0000".join([owner, board_name, runtime_id, name])
        digest = hmac.new(
            self._secret.encode("utf-8"), message.encode("utf-8"), sha256
        ).hexdigest()
        return digest[:32]

    def register(
        self,
        owner: str,
        runtime_id: str,
        service_uuid: str,
        handler: MountHandler,
        # What the mount is called, and the board's, so the address survives a
        # reload. A service that names nothing is identified by its own uuid,
        # which is stable in a board file too.
        board_name: str = "",
        mount_name: str | None = None,
    ) -> MountHandle | None:
        mount_id = self._derive_id(
            owner, board_name, runtime_id, mount_name or service_uuid
        )
        mount_path = f"{MOUNT_PREFIX}/{mount_id}"
        url = self._public_url_for(mount_path)
        if not url:
            return None

        self._mounts[mount_id] = _MountRecord(
            owner=owner,
            runtime_id=runtime_id,
            service_uuid=service_uuid,
            handler=handler,
        )

        def release() -> None:
            self._mounts.pop(mount_id, None)

        return MountHandle(url=url, path=mount_path, release=release)

    def release_runtime(self, owner: str, runtime_id: str) -> None:
        """Drop every mount belonging to a runtime.

        Services release their own mounts on destroy; this is the backstop so a
        torn-down runtime can never leave a publicly reachable endpoint behind.
        """
        for mount_id, record in list(self._mounts.items()):
            if record.owner == owner and record.runtime_id == runtime_id:
                del self._mounts[mount_id]

    def release_owner(self, owner: str) -> None:
        for mount_id, record in list(self._mounts.items()):
            if record.owner == owner:
                del self._mounts[mount_id]

    def count_for_owner(self, owner: str) -> int:
        return sum(1 for r in self._mounts.values() if r.owner == owner)

    async def handle(self, request: web.Request) -> web.StreamResponse:
        """Serve a request addressed to a mount, or 404 when the id is unknown."""
        mount_id = request.match_info.get("mount_id", "")
        record = self._mounts.get(mount_id)
        if not record:
            raise web.HTTPNotFound()

        sub_path = request.match_info.get("sub_path", "")
        if not sub_path.startswith("/"):
            sub_path = "/" + sub_path
        if request.query_string:
            sub_path = f"{sub_path}?{request.query_string}"

        context = MountContext(
            mount_path=f"{MOUNT_PREFIX}/{mount_id}",
            sub_path=sub_path,
        )
        return await record.handler(request, context)


class MountFactory(Protocol):
    def __call__(
        self,
        service_uuid: str,
        handler: MountHandler,
        board_name: str = "",
        mount_name: str | None = None,
    ) -> MountHandle | None: ...


@dataclass
class RuntimeMounts:
    """Grants a runtime's services public endpoints. Supplied by the server,
    which owns the listening socket; absent for inner sub-service pipelines,
    which are not addressable from outside."""

    mount: MountFactory


def decode_body(body: bytes, content_type: str | None) -> Any:
    """Decode a body for the content types where a board would otherwise be
    stuck with raw bytes.

    Returns ``None`` when there is nothing sensible to produce, which includes
    malformed input: a public endpoint receives whatever it is given, and a
    parse failure should leave the raw bytes to inspect rather than fail the
    request.
    """
    import json
    from urllib.parse import parse_qsl

    if not body:
        return None

    media = (content_type or "").split(";")[0].strip().lower()

    if media == "application/json" or media.endswith("+json"):
        try:
            return json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    if media == "application/x-www-form-urlencoded":
        try:
            return dict(parse_qsl(body.decode("utf-8")))
        except UnicodeDecodeError:
            return None

    if media.startswith("text/"):
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError:
            return None

    return None
