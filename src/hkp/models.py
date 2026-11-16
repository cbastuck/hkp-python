"""Models a runtime has loaded, shared by the services that use them.

A board that lays a model out as services names the same weights from every
one of them. Each resolves its asset reference to the same file; this is what
keeps them from each parsing it. A model is loaded by the first service that
asks for it, handed to every later one, and let go of when the last service
holding it releases it.

Keyed by whatever says two loads would produce the same thing — the asset's
``sha256`` and what it was loaded as — never by an asset's id, which a board
may point at other content while it runs.

One per asset store, so a nested pipeline, which asks the runtime around it for
its assets, finds the models there too.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Hashable


class ModelLease:
    """One service's hold on a loaded model."""

    def __init__(self, cache: "ModelCache", key: Hashable, value: Any) -> None:
        self.value = value
        self._cache = cache
        self._key = key
        self._held = True

    def release(self) -> None:
        """Lets go. A second call does nothing, so ``destroy`` can always call it."""
        if self._held:
            self._held = False
            self._cache._release(self._key)


class _Entry:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.loaded = False
        self.value: Any = None
        self.holders = 0
        self.close: Callable[[Any], None] | None = None


class ModelCache:
    def __init__(self) -> None:
        self._entries: dict[Hashable, _Entry] = {}
        self._lock = threading.Lock()

    def acquire(
        self,
        key: Hashable,
        load: Callable[[], Any],
        close: Callable[[Any], None] | None = None,
    ) -> ModelLease:
        """The model under ``key``, loading it when nobody holds it yet.

        Blocking, and for as long as the load takes: two services asking at
        once wait for one load. A load that raises leaves nothing behind, and
        the next to ask tries again.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = self._entries[key] = _Entry()
            entry.holders += 1
        try:
            with entry.lock:
                if not entry.loaded:
                    entry.value = load()
                    entry.close = close
                    entry.loaded = True
        except BaseException:
            self._release(key)
            raise
        return ModelLease(self, key, entry.value)

    def keys(self) -> list[Hashable]:
        """What is loaded, for saying what a runtime holds."""
        with self._lock:
            return [key for key, entry in self._entries.items() if entry.loaded]

    def _release(self, key: Hashable) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.holders -= 1
            if entry.holders > 0:
                return
            del self._entries[key]
        if entry.loaded and entry.close:
            try:
                entry.close(entry.value)
            except Exception as error:  # noqa: BLE001
                print(f"[models] closing {key!r} failed: {error}")
