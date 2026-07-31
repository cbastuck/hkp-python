"""Public service endpoints, served by the shared runtime server.

Mirrors hkp-node's ``src/mounts.ts``.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

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
    """

    def __init__(self, public_url_for: Callable[[str], str | None]) -> None:
        #: Resolves a mount path to the URL clients should use. Returns None
        #: before the server is listening, since the port is not known until then.
        self._public_url_for = public_url_for
        self._mounts: dict[str, _MountRecord] = {}

    def register(
        self,
        owner: str,
        runtime_id: str,
        service_uuid: str,
        handler: MountHandler,
    ) -> MountHandle | None:
        mount_id = secrets.token_hex(16)
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


@dataclass
class RuntimeMounts:
    """Grants a runtime's services public endpoints. Supplied by the server,
    which owns the listening socket; absent for inner sub-service pipelines,
    which are not addressable from outside."""

    mount: Callable[[str, MountHandler], MountHandle | None]


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
