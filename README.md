# hkp-python

asyncio port of the hkp-node runtime engine, built with [aiohttp](https://docs.aiohttp.org/).

## Setup

From the hkp-python directory:

```bash
# 1. Create the venv (or reuse the existing .venv from run_tests.sh)
python3 -m venv .venv

# 2. Install the package with all extras (ASR + dev tooling)
.venv/bin/pip install -r requirements.txt

# 3. Start the runtime (no need to activate the venv)
.venv/bin/hkp-python
```

You should see `hkp-python listening on http://127.0.0.1:8080`.

`requirements.txt` is a thin pointer — the actual dependency list lives in
`pyproject.toml`. To install selectively:

```bash
pip install -e .          # base runtime only
pip install -e ".[asr]"   # + speech-to-text (faster-whisper, numpy)
pip install -e ".[dev]"   # + test tooling
```

## Running the server

```bash
python3 -m hkp
```

The server listens on `0.0.0.0:8080` by default. Configure with environment variables:

| Variable          | Default     | Description                                                   |
| ----------------- | ----------- | ------------------------------------------------------------- |
| `PORT`            | `8080`      | TCP port to listen on                                         |
| `HOST`            | `0.0.0.0`   | Bind address                                                  |
| `EXTERNAL_HOST`   | `127.0.0.1` | Host used in `outputUrl` / WebSocket URLs returned to clients |
| `ALLOWED_ORIGINS` | `*`         | CORS + WebSocket Origin allowlist (comma-separated)           |
| `AUTH0_DOMAIN`    | —           | Auth0 tenant domain; enables JWT auth together with `AUTH0_AUDIENCE` |
| `AUTH0_AUDIENCE`  | —           | Expected `aud` claim (the SPA client id — the frontend sends the ID token) |
| `ALLOWED_EMAILS`  | —           | Comma-separated email allowlist; requires Auth0 config, matched against the **verified** `email` claim |
| `ALLOW_NO_AUTH`   | —           | `true` permits an unauthenticated non-loopback bind — honored only when running from a source checkout, never for a pip-installed package |

Variables may also be placed in a `.env` file in the project root (real
environment variables win; set `SKIP_LOADING_ENV=1` to skip it).

Example:

```bash
PORT=9000 EXTERNAL_HOST=myhost.local python3 -m hkp
```

### Authentication

The auth model mirrors hkp-node exactly, so the same client (hkp-frontend, the
hkp-node coordinator) works against both runtimes:

- With `AUTH0_DOMAIN` + `AUTH0_AUDIENCE` set, every HTTP request needs
  `Authorization: Bearer <token>`; WebSocket upgrades take the token from the
  Authorization header or `?access_token=` (browsers can't set WS headers) and
  are Origin-checked against `ALLOWED_ORIGINS`.
- `POST /runtimes/{id}/session-token` (JWT-gated) mints an opaque in-memory
  token bound to the calling user and that runtime — used by the hkp-node
  coordinator for long-lived machine calls past JWT expiry. Tokens are purged
  when the runtime is removed.
- Fail closed: without Auth0 config the server only starts on a loopback bind
  (`HOST=127.0.0.1`), or from a source checkout with `ALLOW_NO_AUTH=true`.
  Setting `ALLOWED_EMAILS` without Auth0 config refuses to start.

## Running the server using the run_server.sh script

./run_server.sh  
 PORT=9000 ./run_server.sh  
 PORT=9000 EXTERNAL_HOST=myhost.local ./run_server.sh

## HTTP API

The API mirrors hkp-node exactly.

| Method   | Path                                            | Description                                |
| -------- | ----------------------------------------------- | ------------------------------------------ |
| `GET`    | `/runtimes`                                     | List runtimes + service registry           |
| `POST`   | `/runtimes`                                     | Create one or more runtimes                |
| `DELETE` | `/runtimes`                                     | Remove all runtimes                        |
| `GET`    | `/runtimes/{id}`                                | Get a runtime                              |
| `DELETE` | `/runtimes/{id}`                                | Remove a runtime                           |
| `POST`   | `/runtimes/{id}`                                | Process input through the runtime pipeline |
| `POST`   | `/runtimes/{id}/session-token`                  | Mint an opaque coordinator session token   |
| `POST`   | `/runtimes/{id}/rearrange`                      | Reorder services                           |
| `GET`    | `/runtimes/{id}/services`                       | List services                              |
| `POST`   | `/runtimes/{id}/services`                       | Add a service                              |
| `GET`    | `/runtimes/{id}/services/{uuid}`                | Get service state                          |
| `GET`    | `/runtimes/{id}/services/{uuid}/property/{key}` | Get a single state property                |
| `POST`   | `/runtimes/{id}/services/{uuid}`                | Configure a service                        |
| `DELETE` | `/runtimes/{id}/services/{uuid}`                | Remove a service                           |

WebSocket endpoint at `/{runtime_id}` — send `{ "type": "processRuntime", "params": {...} }` to process input; receives `notification` and `result` messages.

## Built-in services

| serviceId                 | Description                                                                |
| ------------------------- | -------------------------------------------------------------------------- |
| `monitor`                 | Logs and re-emits every value it receives                                  |
| `map`                     | Transforms payloads via templates (modes: `overwrite`, `add`, `replace`)   |
| `sub-service`             | Embeds an inner pipeline of services                                       |
| `http-server-subservices` | Runs an embedded HTTP server that drives an inner pipeline on each request |
| `hookup.to/service/timer` | Emits ticks on an interval                                                  |
| `speech-to-text`          | Transcribes 16 kHz mono `FloatRingBuffer` audio with a local Whisper model (requires the `asr` extra) |

## Testing

```bash
./run_tests.sh        # runs all tests
./run_tests.sh -v     # verbose
./run_tests.sh -k map # filter by name
```

The script creates `.venv` automatically if it does not exist.
