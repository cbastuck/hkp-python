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

Content is held in memory, and no asset larger than 64 MB is. What a library
loads by file path — a model's weights — is resolved to a **path** instead
(``resolve_path``): the content is written once to a folder this server owns,
under its ``sha256``, and found there by every runtime and after every restart.
Only an asset that says its ``sha256`` is kept that way; the hash is the file's
name and the check that what was fetched is what the board meant.

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
import os
import re
import tempfile
import threading
import urllib.error
import urllib.request
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
from urllib.parse import urlsplit

from .egress import refusal_in
from .egress import urlopen as egress_urlopen
from .models import ModelCache


ASSET_SCHEME = "hkp-asset://"

_ID = r"[A-Za-z0-9_.\-]+"
_ID_PATTERN = re.compile(rf"^{_ID}$")
_WHOLE_REFERENCE = re.compile(rf"^hkp-asset://({_ID})$")
_ANY_REFERENCE = re.compile(rf"hkp-asset://({_ID})")

DEFAULT_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_CACHE_BYTES = 128 * 1024 * 1024
DEFAULT_TIMEOUT_S = 30.0
#: Everything kept on disk together, and so also the largest single file.
DEFAULT_MAX_FILE_CACHE_BYTES = 8 * 1024 * 1024 * 1024
_CHUNK = 1024 * 1024

#: Told how much of a file has arrived, and how much is expected (None when the
#: source did not say).
Progress = Callable[[int, "int | None"], None]

#: A descriptor as it travels: ``id``, ``mediaType``, exactly one of ``text``,
#: ``base64`` or ``url``, and optional ``name``, ``sha256``, ``size``,
#: ``encoding`` — how the content is stored packed, ``gzip`` being unpacked
#: before it is handed over — and ``runtimes``, the runtimes it is for, which
#: whoever provisions them reads.
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


@dataclass
class AssetPathResolution:
    #: The file holding the content, or None when there is none.
    path: Path | None
    #: Why there is none, or "" when there is.
    problem: str
    #: What the file holds, for keying whatever is loaded from it.
    sha256: str = ""


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
    encoding = value.get("encoding")
    if isinstance(encoding, str) and encoding:
        descriptor["encoding"] = encoding
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


class AssetFileCache:
    """Asset content kept as files, each named by the ``sha256`` of what it holds.

    One folder per server, shared by its runtimes: two boards naming the same
    model fetch it once. A file appears under its name only after its content
    was hashed and matched, so what is found here is not checked again. When
    the folder passes its limit the files used longest ago go first.
    """

    def __init__(self, folder: Path | str, max_bytes: int = DEFAULT_MAX_FILE_CACHE_BYTES) -> None:
        self.folder = Path(folder)
        self.max_bytes = max_bytes
        self._locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()
        #: How far each fetch in flight is: (received, expected).
        self._fetching: dict[str, tuple[int, int | None]] = {}

    def fetching(self, sha256: str) -> tuple[int, int | None] | None:
        """How far a fetch of this file is, for whoever waits on one somebody
        else is doing. None when nobody is fetching it."""
        with self._lock:
            return self._fetching.get(sha256)

    def lock(self, sha256: str) -> threading.Lock:
        """Held while a file is fetched, so two askers wait for one download."""
        with self._lock:
            return self._locks.setdefault(sha256, threading.Lock())

    def get(self, sha256: str) -> Path | None:
        path = self.folder / sha256
        if not path.is_file():
            return None
        try:
            os.utime(path)  # used now, for what goes first when space runs out
        except OSError:
            pass
        return path

    def put(
        self,
        sha256: str,
        chunks: Iterable[bytes],
        *,
        expected: int | None = None,
        progress: Progress | None = None,
    ) -> Path:
        """Writes ``chunks`` to the file named ``sha256``, if that is what they are."""
        self.folder.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        written = 0
        handle, temporary = tempfile.mkstemp(dir=self.folder, prefix=".fetching-")
        try:
            with os.fdopen(handle, "wb") as file:
                for chunk in chunks:
                    written += len(chunk)
                    if written > self.max_bytes:
                        raise _Refused(f"larger than {self.max_bytes} bytes")
                    digest.update(chunk)
                    file.write(chunk)
                    with self._lock:
                        self._fetching[sha256] = (written, expected)
                    if progress:
                        progress(written, expected)
            actual = digest.hexdigest()
            if actual != sha256:
                raise _Refused(f"content does not match its sha256 (got {actual})")
            path = self.folder / sha256
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        finally:
            with self._lock:
                self._fetching.pop(sha256, None)
        self._make_room(keep=path)
        return path

    def _make_room(self, keep: Path) -> None:
        files = []
        for entry in self.folder.iterdir():
            if entry.name.startswith(".") or not entry.is_file():
                continue
            status = entry.stat()
            files.append((status.st_mtime, status.st_size, entry))
        total = sum(size for _, size, _ in files)
        for _, size, entry in sorted(files, key=lambda item: item[0]):
            if total <= self.max_bytes:
                break
            if entry == keep:
                continue
            try:
                # A model loaded from it stays loaded; the next load fetches.
                entry.unlink()
                total -= size
            except OSError:
                pass


_default_file_cache: AssetFileCache | None = None
_default_file_cache_lock = threading.Lock()


def default_file_cache() -> AssetFileCache:
    """This server's folder: ``HKP_ASSET_CACHE``, else ``~/.hkp/python/asset-cache``,
    holding at most ``HKP_ASSET_CACHE_MAX_BYTES``."""
    global _default_file_cache
    with _default_file_cache_lock:
        if _default_file_cache is None:
            folder = os.environ.get("HKP_ASSET_CACHE") or (
                Path.home() / ".hkp" / "python" / "asset-cache"
            )
            limit = os.environ.get("HKP_ASSET_CACHE_MAX_BYTES", "")
            _default_file_cache = AssetFileCache(
                folder, int(limit) if limit.isdigit() and int(limit) > 0 else DEFAULT_MAX_FILE_CACHE_BYTES
            )
        return _default_file_cache


class AssetStore:
    """The descriptors one runtime was given, and the content they resolve to."""

    def __init__(
        self,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_cache_bytes: int = DEFAULT_MAX_CACHE_BYTES,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        opener: Callable[..., Any] | None = None,
        file_cache: AssetFileCache | None = None,
    ) -> None:
        #: Where content resolved to a path is kept; the server's own folder
        #: unless given one. Not touched until something asks for a path.
        self._file_cache = file_cache
        #: What this runtime's services loaded from those files; see models.py.
        self.models = ModelCache()
        self._descriptors: dict[str, AssetDescriptor] = {}
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._cached_bytes = 0
        self._listeners: dict[str, set[Callable[[str], None]]] = {}
        self._max_bytes = max_bytes
        self._max_cache_bytes = max_cache_bytes
        self._timeout_s = timeout_s
        self._open = opener or egress_urlopen
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
        try:
            content = self._unpack(descriptor, content)
        except _Refused as refused:
            return AssetResolution(None, f'asset "{asset_id}": {refused}')

        asset = ResolvedAsset(asset_id, descriptor["mediaType"], content)
        with self._lock:
            # Only if the descriptor is still the one this was loaded for: an edit
            # that arrived while a fetch was in flight must not be shadowed by it.
            if _version(self._descriptors.get(asset_id)) == version:
                self._remember(asset_id, _CacheEntry(version, asset, etag))
        return AssetResolution(asset, "")

    def resolve_path(self, reference: str, progress: Progress | None = None) -> AssetPathResolution:
        """A file holding an asset's content, for a library that loads by path.

        Blocking, and for a first use possibly for minutes: the content is
        fetched to this server's folder, and ``progress`` is told how far it
        is. Every later use, by any runtime, finds the file there.

        Only for an asset that says its ``sha256``. The file belongs to the
        server: it is read, never written to or removed by whoever is handed it.
        """
        asset_id = parse_asset_ref(reference)
        if not asset_id:
            return AssetPathResolution(None, f"{json.dumps(reference)} is not an asset reference")
        with self._lock:
            descriptor = self._descriptors.get(asset_id)
        if descriptor is None:
            return AssetPathResolution(None, f'asset "{asset_id}" is not known to this runtime')
        sha256 = descriptor.get("sha256")
        if not sha256:
            return AssetPathResolution(
                None, f'asset "{asset_id}" does not say its sha256, so it is not kept as a file'
            )
        if descriptor.get("encoding"):
            return AssetPathResolution(
                None,
                f'asset "{asset_id}": content stored as {json.dumps(descriptor["encoding"])} is not kept as a file',
            )

        files = self._file_cache or default_file_cache()
        lock = files.lock(sha256)
        try:
            # Somebody else may be fetching it: wait for them, and say how far
            # they are, so that every service naming a model shows the download
            # and not only the one that happened to start it.
            while not lock.acquire(timeout=0.25):
                under_way = files.fetching(sha256)
                if progress and under_way:
                    progress(*under_way)
        except Exception as error:  # noqa: BLE001 — a caller that stopped waiting
            return AssetPathResolution(None, f'asset "{asset_id}": {error}')
        try:
            found = files.get(sha256)
            if found is not None:
                return AssetPathResolution(found, "", sha256)
            try:
                if "url" in descriptor:
                    expected, chunks = self._stream(descriptor["url"], files.max_bytes)
                else:
                    content, _ = self._load(descriptor, None)
                    expected, chunks = len(content), iter((content,))
                path = files.put(sha256, chunks, expected=expected, progress=progress)
            except _Refused as refused:
                return AssetPathResolution(None, f'asset "{asset_id}": {refused}')
            except Exception as error:  # noqa: BLE001 — reported by name, never raised
                return AssetPathResolution(None, f'asset "{asset_id}": {error}')
        finally:
            lock.release()
        return AssetPathResolution(path, "", sha256)

    # ── Private ──────────────────────────────────────────────────────────────

    def _stream(self, url: str, limit: int) -> tuple[int | None, Iterator[bytes]]:
        """A URL's content in pieces, and how much of it the source says there is."""
        scheme = urlsplit(url).scheme.lower()
        if scheme not in ("http", "https"):
            if scheme == "file":
                raise _Refused("file:// sources cannot be read by this runtime")
            raise _Refused(f"{scheme}:// sources are not supported by this runtime")
        try:
            response = self._open(urllib.request.Request(url), timeout=self._timeout_s)
        except Exception as error:  # noqa: BLE001
            raise _unreachable(url, error) from error
        declared = response.headers.get("Content-Length")
        expected = int(declared) if declared and declared.isdigit() else None
        if expected is not None and expected > limit:
            response.close()
            raise _Refused(f"{url} is larger than {limit} bytes")

        def chunks() -> Iterator[bytes]:
            try:
                with response:
                    while True:
                        chunk = response.read(_CHUNK)
                        if not chunk:
                            return
                        yield chunk
            except _Refused:
                raise
            except Exception as error:  # noqa: BLE001
                raise _unreachable(url, error) from error

        return expected, chunks()

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
            refused = refusal_in(error)
            if refused:
                raise _Refused(f"{url}: {refused}") from error
            raise _Refused(f"{url} is unreachable: {error.reason}") from error
        except TimeoutError as error:
            raise _Refused(f"{url} did not answer in time") from error

    def _unpack(self, descriptor: AssetDescriptor, content: bytes) -> bytes:
        """The content as a service is handed it: unpacked where the descriptor
        says it is stored packed, and held to the same limit as what is fetched."""
        encoding = descriptor.get("encoding")
        if not encoding:
            return content
        if encoding != "gzip":
            raise _Refused(f"content encoded as {json.dumps(encoding)} cannot be unpacked by this runtime")
        try:
            unpacker = zlib.decompressobj(wbits=zlib.MAX_WBITS | 16)
            unpacked = unpacker.decompress(content, self._max_bytes + 1)
        except zlib.error as error:
            raise _Refused("its content is not gzip") from error
        if len(unpacked) > self._max_bytes:
            raise _Refused(f"unpacks to more than {self._max_bytes} bytes")
        if not unpacker.eof:
            raise _Refused("its content is not gzip")
        return unpacked

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


def _unreachable(url: str, error: Exception) -> _Refused:
    """Why a URL gave nothing, in the words ``resolve`` uses for the same."""
    if isinstance(error, _Refused):
        return error
    if isinstance(error, urllib.error.HTTPError):
        return _Refused(f"{url} answered {error.code}")
    if isinstance(error, urllib.error.URLError):
        refused = refusal_in(error)
        if refused:
            return _Refused(f"{url}: {refused}")
        return _Refused(f"{url} is unreachable: {error.reason}")
    if isinstance(error, TimeoutError):
        return _Refused(f"{url} did not answer in time")
    return _Refused(f"{url}: {error}")
