from __future__ import annotations

import asyncio
import os

from .server import create_runtime_server


def _read_integer(value: str | None, fallback: int) -> int:
    if not value:
        return fallback
    try:
        return int(value)
    except ValueError:
        return fallback


async def main() -> None:
    port = _read_integer(os.environ.get("PORT"), 8080)
    host = os.environ.get("HOST", "0.0.0.0")
    external_host = os.environ.get("EXTERNAL_HOST", "127.0.0.1")
    allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")

    server = create_runtime_server(
        {
            "allowed_origins": allowed_origins,
            "external_host": external_host,
            "host": host,
        }
    )

    address = await server.start(port, host)
    print(f"hkp-python listening on {address['base_url']}")

    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        await server.stop()


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
