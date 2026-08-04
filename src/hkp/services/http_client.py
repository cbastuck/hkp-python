from __future__ import annotations

# Service Documentation
# Service ID: http-client
# Service Name: HTTP Client
# Runtime: hkp-python
# Modes: none (method is configuration, not a mode)
# Key Config: url, __hkpMount (target), path, method, headers, userAgent, body
# IO: in=body to send (str | dict | bytes | {meta, body|binary})
#     out=None immediately; the response is pushed through the rest of the
#     pipeline when it arrives, shaped {meta, body?, binary?}
#
# The python implementation of the http-client concept hkp-node and hkp-rt also
# provide, sharing their state contract (url, method, headers, userAgent, body)
# and therefore the same UI panel.
#
# What it adds, like hkp-node's, is __hkpMount, which takes precedence over url
# when set: an address, or a hkp-mount://<runtimeId>/<serviceUuid> reference to
# the service that owns the mount. A reference is resolved by the board's
# coordinator, the only instance that can see across runtimes, and this service
# is configured with the resulting address before it runs. Seeing a reference
# here therefore means the owner has not published an address yet — a normal
# state while a board is still coming up, not an error.
#
# The response shape mirrors what http-server-subservices produces for an
# incoming request, so a pipeline that handles one handles the other.

import asyncio
import json
from typing import Any

import aiohttp

from ..mount import MOUNT_FIELD, is_mount_reference, join_mount_path
from ..types import JsonRecord, NotifyCallback, ServiceConfiguration, ServiceRegistryEntry

HTTP_CLIENT_DESCRIPTOR = ServiceRegistryEntry(
    service_id="http-client",
    service_name="HTTP Client",
    version="v1",
    capabilities=[],
)

# Lower case, as hkp-rt's http-client stores them and the shared UI sends them.
_METHODS = ("get", "post", "put", "patch", "delete")


def _media_type(content_type: str | None) -> str:
    """Content type with any parameters (``; charset=…``) stripped, lower-cased."""
    return (content_type or "").split(";")[0].strip().lower()


def _is_textual(media_type: str) -> bool:
    """Whether a response of this type is worth decoding rather than kept as bytes."""
    return (
        media_type.startswith("text/")
        or media_type == "application/json"
        or media_type.endswith("+json")
        or media_type == "application/x-www-form-urlencoded"
    )


class HttpClientService:
    service_id = HTTP_CLIENT_DESCRIPTOR.service_id
    service_name = HTTP_CLIENT_DESCRIPTOR.service_name
    version = HTTP_CLIENT_DESCRIPTOR.version
    capabilities = HTTP_CLIENT_DESCRIPTOR.capabilities

    def __init__(self, config: ServiceConfiguration, _create_service: Any = None) -> None:
        self.uuid = config.uuid
        self._host: Any = None
        self._url = ""
        self._mount = ""
        self._path = ""
        self._method = "get"
        self._headers: dict[str, str] = {}
        self._user_agent = ""
        self._body = ""
        self._timeout_ms = 10000
        self._bypass = False
        self._in_flight = 0

        if config.state:
            self.configure(config.state)

    def set_host(self, host: Any) -> None:
        self._host = host

    def get_state(self) -> JsonRecord:
        return {
            "url": self._url,
            # Reserved name: the coordinator reads and rewrites it. Holds the
            # address to call, or a reference to the service that owns it while
            # unresolved.
            MOUNT_FIELD: self._mount,
            "path": self._path,
            "method": self._method,
            "headers": self._headers,
            "userAgent": self._user_agent,
            "body": self._body,
            "timeoutMs": self._timeout_ms,
            "bypass": self._bypass,
        }

    def configure(self, config: JsonRecord) -> JsonRecord:
        if isinstance(config.get("url"), str):
            self._url = config["url"]
        if isinstance(config.get(MOUNT_FIELD), str):
            self._mount = config[MOUNT_FIELD]
        if isinstance(config.get("path"), str):
            self._path = config["path"]
        method = config.get("method")
        if isinstance(method, str) and method.lower() in _METHODS:
            self._method = method.lower()
        headers = config.get("headers")
        if isinstance(headers, dict):
            self._headers = {
                key: value for key, value in headers.items() if isinstance(value, str)
            }
        if isinstance(config.get("userAgent"), str):
            self._user_agent = config["userAgent"]
        if isinstance(config.get("body"), str):
            self._body = config["body"]
        timeout = config.get("timeoutMs")
        if isinstance(timeout, (int, float)) and timeout > 0:
            self._timeout_ms = int(timeout)
        if isinstance(config.get("bypass"), bool):
            self._bypass = config["bypass"]
        return self.get_state()

    def process(self, input: Any, notify: NotifyCallback) -> Any:
        """Start the request and stop the synchronous push.

        The runtime calls services one after another without awaiting, so a
        response cannot be returned from here — it does not exist yet. Returning
        None stops the push, and the rest of the pipeline is called with the
        response once it arrives (the inversion-of-control path a cache or a
        fetching service takes).
        """
        if self._bypass:
            return input

        target = self._target_url()
        if not target:
            # Either nothing is configured, or the mount's owner has not
            # published an address yet. Say so and stop; the next input tries
            # again, by which time the coordinator has usually handed it over.
            notify(
                {
                    "error": (
                        f'Waiting for "{self._mount}" to publish an endpoint'
                        if is_mount_reference(self._mount)
                        else "No target configured"
                    )
                }
            )
            return None

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop (e.g. a unit test calling process directly). Nothing
            # can be scheduled, so say so rather than failing silently.
            notify({"error": "No event loop to run the request on"})
            return None

        loop.create_task(self._send(target, input, notify))
        return None

    def destroy(self) -> None:
        self._host = None

    # ── Private ────────────────────────────────────────────────────────────────

    def _target_url(self) -> str | None:
        """The URL to call, or None while there is nothing callable.

        A mount takes precedence over a typed URL — a board that names a service
        is being explicit about which endpoint it means, and the address is not
        knowable when the board is written. An unresolved reference is therefore
        "not ready yet" rather than a reason to fall back to ``url``, which would
        silently call something else.
        """
        if self._mount:
            if is_mount_reference(self._mount):
                return None
            return join_mount_path(self._mount, self._path)
        if not self._url:
            return None
        return join_mount_path(self._url, self._path)

    async def _send(self, url: str, input: Any, notify: NotifyCallback) -> None:
        body, content_type = self._request_body(input)
        headers = dict(self._headers)
        if content_type and "content-type" not in {k.lower() for k in headers}:
            headers["content-type"] = content_type
        if self._user_agent and "user-agent" not in {k.lower() for k in headers}:
            headers["user-agent"] = self._user_agent

        self._in_flight += 1
        notify({"requesting": True, "url": url, "inFlight": self._in_flight})

        try:
            timeout = aiohttp.ClientTimeout(total=self._timeout_ms / 1000)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(
                    self._method.upper(), url, data=body, headers=headers
                ) as response:
                    result = await self._read_response(url, response)
                    notify(
                        {
                            "requesting": False,
                            "url": url,
                            "status": response.status,
                            "inFlight": self._in_flight - 1,
                        }
                    )
            self._push(result, notify)
        except Exception as err:  # noqa: BLE001 - reported, not swallowed
            notify({"requesting": False, "url": url, "error": str(err)})
            # A failed request produces no result to pass on: the pipeline behind
            # this service is not called, rather than called with a fabricated
            # one.
        finally:
            self._in_flight -= 1

    def _request_body(self, input: Any) -> tuple[Any, str | None]:
        """Turn pipeline input into a request body.

        Accepts what an upstream ``http-server-subservices`` produces, so a
        request received on one runtime can be forwarded from another unchanged.
        """
        if self._method == "get":
            return None, None

        if input is None:
            # Nothing came down the pipeline, so send the configured body — which
            # is how the shared UI's body field is meant to be used.
            return (self._body, "text/plain; charset=utf-8") if self._body else (None, None)

        if isinstance(input, str):
            return input, "text/plain; charset=utf-8"
        if isinstance(input, (bytes, bytearray)):
            return bytes(input), "application/octet-stream"

        if isinstance(input, dict):
            meta = input.get("meta")
            declared = meta.get("contentType") if isinstance(meta, dict) else None
            binary = input.get("binary")
            if isinstance(binary, (bytes, bytearray)):
                return bytes(binary), declared or "application/octet-stream"
            if meta is not None and "body" in input:
                body = input["body"]
                if isinstance(body, str):
                    return body, declared or "text/plain; charset=utf-8"
                return json.dumps(body), declared or "application/json"

        return json.dumps(input), "application/json"

    async def _read_response(
        self, url: str, response: aiohttp.ClientResponse
    ) -> JsonRecord:
        """Shape a response the way ``http-server-subservices`` shapes a request:
        metadata always, a decoded body when the content type says what the bytes
        mean, the bytes themselves when it does not.
        """
        content_type = response.headers.get("content-type")
        media_type = _media_type(content_type)
        meta: JsonRecord = {
            "url": url,
            "status": response.status,
            "statusText": response.reason or "",
        }
        if content_type:
            meta["contentType"] = content_type

        raw = await response.read()
        if not raw:
            return {"meta": meta}

        if _is_textual(media_type):
            text = raw.decode("utf-8", errors="replace")
            if media_type == "application/json" or media_type.endswith("+json"):
                try:
                    return {"meta": meta, "body": json.loads(text)}
                except json.JSONDecodeError:
                    # Declared JSON that is not JSON: hand over the text rather
                    # than dropping the response.
                    return {"meta": meta, "body": text}
            return {"meta": meta, "body": text}

        return {"meta": meta, "binary": raw}

    def _push(self, result: JsonRecord, notify: NotifyCallback) -> None:
        """Run the rest of the pipeline with the response, then emit the runtime's
        result. A service that produces data outside the push has to emit it
        itself; nothing else will, and running the remaining services alone would
        leave the chain dead from here on.
        """
        if not self._host:
            return
        output = self._host.process_from(
            self.uuid,
            result,
            # No-op: the runtime already fans these out to its notification
            # targets. Re-notifying through the host would deliver each twice.
            lambda _n: None,
        )
        # A downstream service returning None means "stop" — honour it rather
        # than forwarding a dead result to the next runtime.
        if output is not None:
            self._host.emit_result(output)
