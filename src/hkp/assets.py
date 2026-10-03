"""Assets a runtime was given, and the one way to their content.

A board declares content once — a page, a script, an image, a model — as an
asset *descriptor*: an id, a media type and exactly one source. Service state
names one by reference, ``hkp-asset://<id>``, as a whole field, and never holds
the content itself. ``get_state`` therefore echoes the reference, and saving a
board writes back what was configured: there is no round trip to undo.

The descriptors arrive with the runtime's create payload, or on
``POST /runtimes/<id>/assets`` — and again whenever an asset is edited, which is
the point: a service resolves its reference at the moment it uses it, so the
next use gets the new content without anything being reconfigured. A runtime is
given every asset of its board that is not kept to other runtimes, named by its
services or not: which one a service uses can be decided as it runs.

Where the content comes from depends on the source:

``text``, ``base64``
    already in the descriptor
``http(s)://``
    fetched by this runtime as anyone would fetch it, cached by ``sha256`` or
    revalidated by ETag

Anything else — ``file://`` among them, since this runtime keeps no files a
board may name — is refused by name rather than guessed at.

An asset carries no request headers and names no secret. It is resolved without
anyone looking, by every runtime holding it, which is no place for a
credential; content that needs one is fetched by a service that says where it
sends it.

The format matches ``hkp-frontend/src/runtime/board/assets.ts`` and
``hkp-node/src/assets.ts``: a board written against one runtime has to open
against another.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import threading
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlsplit


ASSET_SCHEME = "hkp-asset://"

_ID = r"[A-Za-z0-9_.\-]+"
_ID_PATTERN = re.compile(rf"^{_ID}$")
_WHOLE_REFERENCE = re.compile(rf"^hkp-asset://({_ID})$")
_ANY_REFERENCE = re.compile(rf"hkp-asset://({_ID})")

DEFAULT_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_CACHE_BYTES = 128 * 1024 * 1024
DEFAULT_TIMEOUT_S = 30.0

#: A descriptor as it travels: ``id``, ``mediaType``, exactly one of ``text``,
#: ``base64`` or ``url``, and optional ``name``, ``sha256``, ``size`` and
#: ``runtimes`` — the runtimes it is for, which whoever provisions them reads.
AssetDescriptor = dict[str, Any]


@dataclass
class ResolvedAsset:
    id: str
    media_type: str
    content: bytes


@dataclass
class AssetResolution:
    #: The content, or None when there is none.
    asset: ResolvedAsset | None
    #: Why there is none, or "" when there is.
    problem: str


def parse_asset_ref(value: Any) -> str | None:
    """The id a whole-value reference names, or None for anything else.

    A reference inside a longer string is not one: nothing is spliced into text.
    """
    if not isinstance(value, str):
        return None
    match = _WHOLE_REFERENCE.match(value)
    return match.group(1) if match else None


def format_asset_ref(asset_id: str) -> str:
    return f"{ASSET_SCHEME}{asset_id}"


def referenced_assets(value: Any) -> list[str]:
    """Every asset id a value mentions, however deeply it is nested.

    Found anywhere in a string, not only as a whole value, so that a reference an
    expression produces is still one this runtime is told about. Finding is
    generous; resolving is not.
    """
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            for asset_id in _ANY_REFERENCE.findall(node):
                if asset_id not in found:
                    found.append(asset_id)
        elif isinstance(node, dict):
            for child in node.values():
                walk(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(value)
    return found


def is_text_media_type(media_type: str) -> bool:
    """Whether a media type is text a pipeline can carry as a string."""
    kind = media_type.split(";")[0].strip().lower()
    return (
        kind.startswith("text/")
        or kind == "application/json"
        or kind.endswith("+json")
        or kind == "application/javascript"
        or kind == "application/xml"
        or kind.endswith("+xml")
    )


def read_asset_descriptor(value: Any, fallback_id: str | None = None) -> AssetDescriptor | None:
    """One descriptor off the wire, or None when it is not one.

    Exactly one source, or it is not an asset: two would leave a runtime choosing
    between them.
    """
    if not isinstance(value, dict):
        return None
    asset_id = value.get("id") if isinstance(value.get("id"), str) else fallback_id
    if not asset_id or not _ID_PATTERN.match(asset_id):
        return None
    sources = [key for key in ("text", "base64", "url") if isinstance(value.get(key), str)]
    if len(sources) != 1:
        return None

    media_type = value.get("mediaType")
    descriptor: AssetDescriptor = {
        "id": asset_id,
        "mediaType": media_type if isinstance(media_type, str) and media_type else "application/octet-stream",
    }
    if isinstance(value.get("name"), str):
        descriptor["name"] = value["name"]
    sha = value.get("sha256")
    if isinstance(sha, str) and re.fullmatch(r"[0-9a-fA-F]{64}", sha):
        descriptor["sha256"] = sha.lower()
    size = value.get("size")
    if isinstance(size, (int, float)) and not isinstance(size, bool):
        descriptor["size"] = size
    runtimes = value.get("runtimes")
    if isinstance(runtimes, list):
        descriptor["runtimes"] = [entry for entry in runtimes if isinstance(entry, str)]
    source = sources[0]
    descriptor[source] = value[source]
    return descriptor


def read_assets_payload(value: Any) -> dict[str, AssetDescriptor | None]:
    """Reads an assets payload off the wire.

    A map of id to descriptor, where ``None`` removes one, or a list of
    descriptors. Anything it cannot read is dropped rather than failing the
    request — a malformed entry costs one asset, and the service referencing it
    says so by name.
    """
    entries: dict[str, AssetDescriptor | None] = {}
    if isinstance(value, list):
        for item in value:
            descriptor = read_asset_descriptor(item)
            if descriptor:
                entries[descriptor["id"]] = descriptor
        return entries
    if not isinstance(value, dict):
        return entries
    for asset_id, item in value.items():
        if item is None:
            if _ID_PATTERN.match(asset_id):
                entries[asset_id] = None
            continue
        descriptor = read_asset_descriptor(item, asset_id)
        if descriptor:
            # Keyed by what the payload named, so the id inside cannot disagree.
            entries[asset_id] = {**descriptor, "id": asset_id}
    return entries


def _version(descriptor: AssetDescriptor | None) -> str:
    return json.dumps(descriptor, sort_keys=True)


@dataclass
class _CacheEntry:
    version: str
    asset: ResolvedAsset
    etag: str | None = None


_BASE64_WHITESPACE = re.compile(r"[\t\n\f\r ]+")
_BASE64_ALPHABET = re.compile(r"[A-Za-z0-9+/]*")


def decode_base64(value: str) -> bytes | None:
    """The bytes base64 text stands for, or ``None`` when it is not base64.

    ``b64decode`` without ``validate`` skips what it does not recognise and
    decodes the rest, so text that is not base64 would come out as some other
    bytes. What is taken here is what a browser's ``atob`` takes — ASCII
    whitespace ignored, the standard alphabet, padding optional — so that a
    descriptor resolves to the same content, or the same refusal, on every
    runtime.
    """
    text = _BASE64_WHITESPACE.sub("", value)
    if len(text) % 4 == 0:
        text = re.sub(r"={1,2}$", "", text)
    if len(text) % 4 == 1 or not _BASE64_ALPHABET.fullmatch(text):
        return None
    try:
        return base64.b64decode(text + "=" * (-len(text) % 4), validate=True)
    except (binascii.Error, ValueError):
        return None


class AssetStore:
    """The descriptors one runtime was given, and the content they resolve to."""

    def __init__(
        self,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_cache_bytes: int = DEFAULT_MAX_CACHE_BYTES,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self._descriptors: dict[str, AssetDescriptor] = {}
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._cached_bytes = 0
        self._listeners: dict[str, set[Callable[[str], None]]] = {}
        self._max_bytes = max_bytes
        self._max_cache_bytes = max_cache_bytes
        self._timeout_s = timeout_s
        self._open = opener or urllib.request.urlopen
        # Services resolve on worker threads and the server pushes on its loop.
        self._lock = threading.RLock()

    def replace(self, entries: dict[str, AssetDescriptor]) -> None:
        """Replaces everything held."""
        with self._lock:
            previous = self._descriptors
            self._descriptors = dict(entries)
            touched = set(previous) | set(self._descriptors)
        for asset_id in touched:
            if _version(previous.get(asset_id)) != _version(self._descriptors.get(asset_id)):
                self._changed(asset_id)

    def merge(self, entries: dict[str, AssetDescriptor | None]) -> None:
        """Adds, replaces or removes (``None``) individual descriptors.

        An asset deleted from the board is gone from every runtime that was told
        about it, rather than served on from a stale copy.
        """
        for asset_id, entry in entries.items():
            with self._lock:
                previous = self._descriptors.get(asset_id)
                if entry is None:
                    self._descriptors.pop(asset_id, None)
                else:
                    self._descriptors[asset_id] = entry
            if _version(previous) != _version(entry):
                self._changed(asset_id)

    def ids(self) -> list[str]:
        """The ids held, for saying what a runtime knows about."""
        with self._lock:
            return list(self._descriptors)

    def descriptor(self, asset_id: str) -> AssetDescriptor | None:
        with self._lock:
            return self._descriptors.get(asset_id)

    def subscribe(self, asset_id: str, listener: Callable[[str], None]) -> Callable[[], None]:
        """Calls ``listener`` with the id whenever the asset changes or goes away.

        Only a service that loads something once needs it — one that resolves on
        every use already gets the new content.
        """
        with self._lock:
            self._listeners.setdefault(asset_id, set()).add(listener)

        def unsubscribe() -> None:
            with self._lock:
                listeners = self._listeners.get(asset_id)
                if listeners is not None:
                    listeners.discard(listener)
                    if not listeners:
                        self._listeners.pop(asset_id, None)

        return unsubscribe

    def resolve(self, reference: str) -> AssetResolution:
        """An asset's content for one use, or a sentence saying why there is none.

        Blocking: a URL source is fetched on the calling thread. A caller on the
        server's loop runs this in a thread.
        """
        asset_id = parse_asset_ref(reference)
        if not asset_id:
            return AssetResolution(None, f"{json.dumps(reference)} is not an asset reference")
        with self._lock:
            descriptor = self._descriptors.get(asset_id)
            cached = self._cache.get(asset_id)
        if descriptor is None:
            return AssetResolution(None, f'asset "{asset_id}" is not known to this runtime')

        version = _version(descriptor)
        revalidate = "url" in descriptor and not descriptor.get("sha256")
        if cached and cached.version == version and not revalidate:
            with self._lock:
                self._cache.move_to_end(asset_id)
            return AssetResolution(cached.asset, "")

        try:
            content, etag = self._load(descriptor, cached if cached and cached.version == version else None)
        except _Refused as refused:
            return AssetResolution(None, f'asset "{asset_id}": {refused}')
        except Exception as error:  # noqa: BLE001 — reported by name, never raised
            return AssetResolution(None, f'asset "{asset_id}": {error}')

        problem = self._check(descriptor, content)
        if problem:
            return AssetResolution(None, f'asset "{asset_id}": {problem}')

        asset = ResolvedAsset(asset_id, descriptor["mediaType"], content)
        with self._lock:
            # Only if the descriptor is still the one this was loaded for: an edit
            # that arrived while a fetch was in flight must not be shadowed by it.
            if _version(self._descriptors.get(asset_id)) == version:
                self._remember(asset_id, _CacheEntry(version, asset, etag))
        return AssetResolution(asset, "")

    # ── Private ──────────────────────────────────────────────────────────────

    def _load(self, descriptor: AssetDescriptor, cached: _CacheEntry | None) -> tuple[bytes, str | None]:
        if "text" in descriptor:
            return descriptor["text"].encode("utf-8"), None
        if "base64" in descriptor:
            content = decode_base64(descriptor["base64"])
            if content is None:
                raise _Refused("its content is not base64")
            return content, None

        url = descriptor["url"]
        scheme = urlsplit(url).scheme.lower()
        if scheme not in ("http", "https"):
            if scheme == "file":
                raise _Refused("file:// sources cannot be read by this runtime")
            raise _Refused(f"{scheme}:// sources are not supported by this runtime")

        headers = {"If-None-Match": cached.etag} if cached and cached.etag else {}

        request = urllib.request.Request(url, headers=headers)
        try:
            with self._open(request, timeout=self._timeout_s) as response:
                declared = response.headers.get("Content-Length")
                if declared and declared.isdigit() and int(declared) > self._max_bytes:
                    raise _Refused(f"{url} is larger than {self._max_bytes} bytes")
                content = response.read(self._max_bytes + 1)
                if len(content) > self._max_bytes:
                    raise _Refused(f"{url} is larger than {self._max_bytes} bytes")
                return content, response.headers.get("ETag")
        except urllib.error.HTTPError as error:
            if error.code == 304 and cached:
                return cached.asset.content, cached.etag
            raise _Refused(f"{url} answered {error.code}") from error
        except urllib.error.URLError as error:
            raise _Refused(f"{url} is unreachable: {error.reason}") from error
        except TimeoutError as error:
            raise _Refused(f"{url} did not answer in time") from error

    def _check(self, descriptor: AssetDescriptor, content: bytes) -> str:
        if len(content) > self._max_bytes:
            return f"larger than {self._max_bytes} bytes"
        expected = descriptor.get("sha256")
        if expected:
            actual = hashlib.sha256(content).hexdigest()
            if actual != expected.lower():
                return f"content does not match its sha256 (got {actual})"
        return ""

    def _remember(self, asset_id: str, entry: _CacheEntry) -> None:
        self._forget(asset_id)
        size = len(entry.asset.content)
        if size > self._max_cache_bytes:
            return
        while self._cache and self._cached_bytes + size > self._max_cache_bytes:
            oldest = next(iter(self._cache))
            self._forget(oldest)
        self._cache[asset_id] = entry
        self._cached_bytes += size

    def _forget(self, asset_id: str) -> None:
        entry = self._cache.pop(asset_id, None)
        if entry is not None:
            self._cached_bytes -= len(entry.asset.content)

    def _changed(self, asset_id: str) -> None:
        with self._lock:
            self._forget(asset_id)
            listeners = list(self._listeners.get(asset_id, ()))
        for listener in listeners:
            try:
                listener(asset_id)
            except Exception as error:  # noqa: BLE001
                print(f'[assets] listener for "{asset_id}" failed: {error}')


class _Refused(Exception):
    """A source that cannot be read, said in a sentence."""
