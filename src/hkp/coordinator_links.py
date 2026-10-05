"""This runtime server's connections to the coordinators its runtimes belong to.

A coordinator never dials a runtime server. When a person deploys a board, their
client — the one party holding a session with both sides — asks the coordinator
for a ticket and tells this server: *connect to that coordinator, with this
ticket*. From then on the runtime the ticket speaks for is built, configured and
driven over the connection this server opened.

Nothing needs to be able to reach this server for that to work, which is why a
laptop behind NAT and a loopback address are not special cases.

The ticket is kept beside this server's other data and presented again after a
restart or a dropped connection, with nobody present. It is all that is kept:
the runtime itself is rebuilt by the coordinator, from the board's config, once
this server is connected again.

The protocol is hkp-node's (`hkp-node/src/coordinator/participantProtocol.ts`),
which is where the coordinator lives.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol
from urllib.parse import urlsplit, urlunsplit

import aiohttp

from .binary_frame import decode_frame, encode_frame, from_binary, to_binary
from .secrets import SecretEntry

#: Where a runtime server connects in; relative to the coordinator's base.
JOIN_PATH = "/join"

#: Close codes that end a link for good. Anything else — a network drop, a
#: coordinator restarting — is something a link reconnects through.
CLOSE_TICKET_REVOKED = 4403
CLOSE_REPLACED = 4409


@dataclass(frozen=True)
class LinkRecord:
    """What is remembered about one link, and everything a reconnect needs."""

    #: The tenant this link acts as: whoever introduced it.
    owner: str
    board_name: str
    runtime_id: str
    coordinator_url: str
    ticket: str


class LinkStore(Protocol):
    def load(self) -> list[LinkRecord]: ...

    def save(self, records: list[LinkRecord]) -> None: ...


class MemoryLinkStore:
    def __init__(self) -> None:
        self._held: list[LinkRecord] = []

    def load(self) -> list[LinkRecord]:
        return list(self._held)

    def save(self, records: list[LinkRecord]) -> None:
        self._held = list(records)


class FileLinkStore:
    """Links kept in one file, readable by its owner only: a ticket is a bearer
    credential for one runtime of one board."""

    def __init__(self, file: str | Path) -> None:
        self._file = Path(file)

    def load(self) -> list[LinkRecord]:
        try:
            parsed = json.loads(self._file.read_text())
        except (OSError, ValueError):
            # Not written yet, or unreadable: either way there is nothing to
            # reconnect with, and the next introduction writes it afresh.
            return []
        if not isinstance(parsed, list):
            return []
        records = []
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            try:
                record = LinkRecord(**entry)
            except TypeError:
                continue
            if all(isinstance(value, str) for value in asdict(record).values()):
                records.append(record)
        return records

    def save(self, records: list[LinkRecord]) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = self._file.with_name(f"{self._file.name}.{os.getpid()}.tmp")
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
        )
        with os.fdopen(descriptor, "w") as handle:
            json.dump([asdict(record) for record in records], handle, indent=2)
        os.replace(temporary, self._file)


class LinkHost(Protocol):
    """What a link may do on this server — exactly the runtime it speaks for,
    as the tenant that introduced it. Supplied by the server, which owns
    runtimes.

    The runtime is the board's: it shares its id with nothing a client created
    here, nor with another board's runtime of the same id. Each operation names
    it by tenant, board and id.
    """

    kind: str

    def link_registry(self) -> list[Any]: ...

    def link_runtime_exists(
        self, owner: str, board_name: str, runtime_id: str
    ) -> bool: ...

    def link_provision(
        self,
        owner: str,
        board_name: str,
        runtime_id: str,
        payload: dict[str, Any],
        secrets: dict[str, SecretEntry],
    ) -> dict[str, Any]: ...

    def link_describe(
        self, owner: str, board_name: str, runtime_id: str
    ) -> dict[str, Any] | None: ...

    async def link_configure_service(
        self,
        owner: str,
        board_name: str,
        runtime_id: str,
        service_uuid: str,
        config: Any,
    ) -> Any: ...

    def link_set_state(
        self, owner: str, board_name: str, runtime_id: str, state: dict[str, Any]
    ) -> Any: ...

    def link_remove(self, owner: str, board_name: str, runtime_id: str) -> None: ...

    async def link_process(
        self, owner: str, board_name: str, runtime_id: str, params: Any, context: Any
    ) -> dict[str, Any]:
        """Runs the runtime's pipeline as the run ``context`` names. Answers
        with the ``result`` message to send: what it produced, and the run it
        produced it in."""
        ...

    def link_process_service(
        self,
        owner: str,
        board_name: str,
        runtime_id: str,
        service_uuid: str,
        params: Any,
        context: Any,
        done: Callable[[dict[str, Any]], None],
    ) -> None:
        """Begins the runtime's pipeline at one service, as the run ``context``
        names. Returns once the work is taken — raising when it cannot be —
        and hands ``done`` the ``result`` message when the pipeline finishes."""
        ...


def join_url_for(coordinator_url: str) -> str:
    """The coordinator's join endpoint for a coordinator's base address."""
    parts = urlsplit(coordinator_url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"Not a coordinator address: {coordinator_url}")
    scheme = "wss" if parts.scheme == "https" else "ws"
    return urlunsplit(
        (scheme, parts.netloc, f"{parts.path.rstrip('/')}{JOIN_PATH}", "", "")
    )


# A runtime id is unique within a board, and a board's name within a tenant.
# NUL occurs in none of the parts.
def _link_key(owner: str, board_name: str, runtime_id: str) -> str:
    return f"{owner}\x00{board_name}\x00{runtime_id}"


class _Link:
    def __init__(
        self,
        record: LinkRecord,
        host: LinkHost,
        options: dict[str, float],
        on_rejected: Callable[["_Link"], None],
        dumps: Callable[[Any], str],
    ) -> None:
        self.record = record
        #: Values for the references this runtime's services carry, handed over
        #: by the person's client when it introduced this link. Held in memory
        #: only: after this server restarts the runtime is rebuilt without
        #: them, and says which ones it is missing.
        self.secrets: dict[str, SecretEntry] = {}
        self._host = host
        self._options = options
        self._on_rejected = on_rejected
        self._dumps = dumps
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._welcomed = False
        self._disposed = False
        self._task: asyncio.Task[None] | None = None
        self._pending: set[asyncio.Task[Any]] = set()
        self._first: asyncio.Future[str | None] | None = None

    @property
    def _runtime(self) -> tuple[str, str, str]:
        """The runtime this link speaks for, as a host's operations name it."""
        return (self.record.owner, self.record.board_name, self.record.runtime_id)

    @property
    def connected(self) -> bool:
        return self._welcomed and self._ws is not None and not self._ws.closed

    def start(self) -> "asyncio.Future[str | None]":
        """Connects, and keeps connected. The future resolves with how the
        first attempt went: None when welcomed, else the reason it was not."""
        loop = asyncio.get_running_loop()
        self._first = loop.create_future()
        self._task = loop.create_task(self._run())
        return self._first

    def _settle_first(self, reason: str | None) -> None:
        if self._first is not None and not self._first.done():
            self._first.set_result(reason)

    async def _run(self) -> None:
        attempts = 0
        while not self._disposed:
            rejected = await self._connect_once()
            if self._disposed:
                return
            if rejected:
                self._disposed = True
                self._on_rejected(self)
                return
            if self._welcomed_once:
                attempts = 0
                self._welcomed_once = False
            delay = min(
                self._options["max_reconnect_delay"],
                self._options["reconnect_delay"] * (2**attempts),
            )
            attempts += 1
            await asyncio.sleep(delay)

    _welcomed_once = False

    async def _connect_once(self) -> bool:
        """One connection, from handshake to close. Returns whether the
        coordinator ended it for good."""
        session = aiohttp.ClientSession()
        try:
            async with session.ws_connect(
                join_url_for(self.record.coordinator_url),
                # A header rather than the URL, which is what ends up in logs.
                headers={"Authorization": f"Bearer {self.record.ticket}"},
                heartbeat=30,
                # No ceiling of the library's own: exceeding one closes the
                # link, and the board would lose its runtime over one large
                # value.
                max_msg_size=0,
            ) as ws:
                self._ws = ws
                await ws.send_str(
                    self._dumps(
                        {
                            "type": "hello",
                            "server": self._host.kind,
                            "registry": self._host.link_registry(),
                            "runtimeExists": self._host.link_runtime_exists(
                                *self._runtime
                            ),
                        }
                    )
                )
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.BINARY:
                        # Input for the pipeline that holds bytes; see
                        # binary_frame.
                        frame = decode_frame(message.data)
                        if frame is None or frame[0].get("type") != "processRuntime":
                            continue
                        header, shape, payload = frame
                        data = {**header, "params": from_binary(shape, payload)}
                    elif message.type == aiohttp.WSMsgType.TEXT:
                        try:
                            data = json.loads(message.data)
                        except ValueError:
                            continue
                        if not isinstance(data, dict):
                            continue
                    else:
                        continue
                    # Each in its own task: a pipeline that runs for a while
                    # must not stop the connection from being read.
                    task = asyncio.ensure_future(self._on_message(ws, data))
                    self._pending.add(task)
                    task.add_done_callback(self._pending.discard)
                code = ws.close_code
                self._settle_first(f"connection closed ({code})")
                # Revoked, or another runtime server took this runtime's place
                # in the board. Either way this server's copy is no longer the
                # board's, and reconnecting would only fight whoever holds the
                # place now.
                return code in (CLOSE_TICKET_REVOKED, CLOSE_REPLACED)
        except aiohttp.WSServerHandshakeError as err:
            # 401 is the coordinator's answer to a ticket it does not hold —
            # replaced, or belonging to a board that was deleted.
            if err.status in (401, 403):
                self._settle_first("the coordinator did not accept the ticket")
                return True
            self._settle_first(f"the coordinator answered {err.status}")
            return False
        except (aiohttp.ClientError, OSError, asyncio.TimeoutError) as err:
            self._settle_first(str(err) or type(err).__name__)
            return False
        finally:
            self._ws = None
            self._welcomed = False
            await session.close()

    async def dispose(self) -> None:
        """Stops for good, without telling anyone."""
        self._disposed = True
        self._settle_first("the link was dropped")
        ws = self._ws
        if ws is not None and not ws.closed:
            await ws.close()
        task = self._task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def _send(self, ws: aiohttp.ClientWebSocketResponse, message: Any) -> None:
        if ws.closed:
            return
        # A result holding bytes goes as a binary frame; as text it could only
        # be described, not carried.
        binary = (
            to_binary(message.get("data"))
            if isinstance(message, dict) and message.get("type") == "result"
            else None
        )
        try:
            if binary is not None:
                # The header is the message without its value, so the run it
                # belongs to travels in it either way.
                header: dict[str, Any] = {"type": "result"}
                if message.get("context") is not None:
                    header["context"] = message["context"]
                await ws.send_bytes(encode_frame(header, *binary))
            else:
                await ws.send_str(self._dumps(message))
        except (ConnectionError, aiohttp.ClientError):
            pass

    def _emit_later(self, message: dict[str, Any]) -> None:
        """Sends what a pipeline finished with, from the loop it finished on."""
        task = asyncio.get_running_loop().create_task(self.emit(message))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def emit(self, message: dict[str, Any]) -> None:
        """The runtime said something; only a welcomed link has anyone to tell."""
        ws = self._ws
        if self._welcomed and ws is not None:
            await self._send(ws, message)

    async def _on_message(
        self, ws: aiohttp.ClientWebSocketResponse, message: dict[str, Any]
    ) -> None:
        runtime = self._runtime
        kind = message.get("type")

        if kind == "welcome":
            self._welcomed = True
            self._welcomed_once = True
            self._settle_first(None)
            return

        if kind == "processRuntime":
            if "params" not in message:
                return
            try:
                result = await self._host.link_process(
                    *runtime, message["params"], message.get("context")
                )
            except Exception as err:  # noqa: BLE001 - reported, never raised
                print(
                    f'[coordinator-link] Runtime "{runtime[2]}" failed to process: {err}'
                )
                return
            await self._send(ws, result)
            return

        if kind == "request":
            request_id = message.get("requestId")
            try:
                data = await self._serve(message)
                await self._send(
                    ws,
                    {
                        "type": "response",
                        "requestId": request_id,
                        "ok": True,
                        "data": data,
                    },
                )
            except Exception as err:  # noqa: BLE001 - the answer is the error
                await self._send(
                    ws,
                    {
                        "type": "response",
                        "requestId": request_id,
                        "ok": False,
                        "error": str(err) or type(err).__name__,
                    },
                )

    async def _serve(self, request: dict[str, Any]) -> Any:
        runtime = self._runtime
        op = request.get("op")
        if op == "provision":
            return self._host.link_provision(
                *runtime,
                # The board this link was introduced for, whatever the request
                # says: a ticket speaks for one board.
                {**request, "boardName": self.record.board_name},
                self.secrets,
            )
        if op == "describe":
            described = self._host.link_describe(*runtime)
            if described is None:
                raise RuntimeError("the runtime is not running")
            return described
        if op == "configureService":
            return await self._host.link_configure_service(
                *runtime, str(request.get("serviceUuid")), request.get("config")
            )
        if op == "processService":
            self._host.link_process_service(
                *runtime,
                str(request.get("serviceUuid")),
                request.get("params"),
                request.get("context"),
                # What the pipeline made of it goes to the coordinator the way
                # any of this runtime's output does.
                self._emit_later,
            )
            return {"accepted": True}
        if op == "setState":
            state = request.get("state")
            return self._host.link_set_state(
                *runtime, state if isinstance(state, dict) else {}
            )
        if op == "remove":
            self._host.link_remove(*runtime)
            return {}
        raise RuntimeError(f'Unknown operation "{op}"')


class CoordinatorLinks:
    def __init__(
        self,
        host: LinkHost,
        store: LinkStore | None = None,
        options: dict[str, float] | None = None,
        *,
        spawn: Callable[[Awaitable[Any]], None],
        dumps: Callable[[Any], str] = json.dumps,
    ) -> None:
        self._host = host
        self._store: LinkStore = store or MemoryLinkStore()
        self._spawn = spawn
        self._dumps = dumps
        self._links: dict[str, _Link] = {}
        options = options or {}
        self._options = {
            #: First delay before reconnecting; doubles up to the maximum.
            "reconnect_delay": options.get("reconnect_delay", 1.0),
            "max_reconnect_delay": options.get("max_reconnect_delay", 30.0),
            #: How long an introduction waits to be welcomed.
            "introduce_timeout": options.get("introduce_timeout", 10.0),
        }

    async def introduce(
        self, record: LinkRecord, secrets: dict[str, SecretEntry] | None = None
    ) -> None:
        """Connects this server to a coordinator as one runtime of one board.

        Returns once the coordinator has accepted the ticket, and raises — and
        keeps nothing — when it has not: an introduction is made by somebody
        waiting to hear whether it worked, so this is the one connection attempt
        that is not retried.

        A runtime that is already connected to that coordinator stays as it is:
        the link it has is kept, the ticket handed over goes unused, and only
        the secrets are taken. Being introduced is the first step of a deploy
        that may yet fail, and must not change what is running. One connected
        to a *different* coordinator is refused — a board's runtime here
        belongs to one coordinator at a time, and the other is named.
        """
        # Validated before anything is replaced: a malformed address must not
        # cost a runtime the link it already has.
        join_url = join_url_for(record.coordinator_url)

        key = _link_key(record.owner, record.board_name, record.runtime_id)
        existing = self._links.get(key)
        if existing is not None and existing.connected:
            if join_url_for(existing.record.coordinator_url) != join_url:
                raise ConnectionError(
                    f'"{record.board_name}" is already deployed here by '
                    f"{existing.record.coordinator_url}"
                )
            existing.secrets = dict(secrets or {})
            return
        previous = self._links.pop(key, None)
        if previous is not None:
            await previous.dispose()

        link = self._create_link(record)
        link.secrets = dict(secrets or {})
        self._links[key] = link

        try:
            reason = await asyncio.wait_for(
                asyncio.shield(link.start()), self._options["introduce_timeout"]
            )
        except asyncio.TimeoutError:
            reason = "the coordinator did not answer"

        if reason is not None:
            await link.dispose()
            if self._links.get(key) is link:
                del self._links[key]
            self._persist()
            raise ConnectionError(reason)
        self._persist()

    def restore(self) -> None:
        """Reconnects with the tickets kept from before this process started."""
        for record in self._store.load():
            key = _link_key(record.owner, record.board_name, record.runtime_id)
            if key in self._links:
                continue
            link = self._create_link(record)
            self._links[key] = link
            link.start()

    def list(self, owner: str) -> list[dict[str, Any]]:
        """A tenant's links, without their tickets.

        ``running`` is whether the runtime the link is for has been built: a
        board's runtimes are not among those a client lists, so this is where
        they are seen.
        """
        return [
            {
                "boardName": link.record.board_name,
                "runtimeId": link.record.runtime_id,
                "coordinatorUrl": link.record.coordinator_url,
                "connected": link.connected,
                "running": self._host.link_runtime_exists(
                    link.record.owner, link.record.board_name, link.record.runtime_id
                ),
            }
            for link in self._links.values()
            if link.record.owner == owner
        ]

    async def remove(self, owner: str, board_name: str, runtime_id: str) -> bool:
        """Leaves a board: drops the link and the runtime it was for."""
        link = self._links.pop(_link_key(owner, board_name, runtime_id), None)
        if link is None:
            return False
        await link.dispose()
        self._host.link_remove(owner, board_name, runtime_id)
        self._persist()
        return True

    def emit(
        self, owner: str, board_name: str, runtime_id: str, message: dict[str, Any]
    ) -> None:
        """Carries a runtime's output to its coordinator, when it has one.

        Safe from a worker thread: a pipeline runs off the event loop, and what
        it says is handed back to it.
        """
        link = self._links.get(_link_key(owner, board_name, runtime_id))
        if link is not None and link.connected:
            self._spawn(link.emit(message))

    async def stop(self) -> None:
        """Closes every connection and keeps every ticket."""
        links = list(self._links.values())
        self._links.clear()
        for link in links:
            await link.dispose()

    def _create_link(self, record: LinkRecord) -> _Link:
        def on_rejected(rejected: _Link) -> None:
            # The coordinator no longer holds this ticket, so the runtime it was
            # for is nobody's: it was built to outlive its clients, and the only
            # party that would have released it has just said it is not theirs.
            key = _link_key(record.owner, record.board_name, record.runtime_id)
            if self._links.get(key) is not rejected:
                return
            del self._links[key]
            self._host.link_remove(record.owner, record.board_name, record.runtime_id)
            self._persist()

        return _Link(record, self._host, self._options, on_rejected, self._dumps)

    def _persist(self) -> None:
        try:
            self._store.save([link.record for link in self._links.values()])
        except OSError as err:
            print(
                "[coordinator-link] Could not persist coordinator links, so they "
                f"will not be re-established after a restart: {err}"
            )
