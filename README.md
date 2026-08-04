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
| `AUTH0_AUDIENCE`  | —           | Accepted `aud` claims, comma-separated — the client id of each Auth0 application whose users this runtime serves (the frontend sends the ID token) |
| `ALLOWED_EMAILS`  | —           | Comma-separated email allowlist; requires Auth0 config, matched against the **verified** `email` claim |
| `ALLOW_NO_AUTH`   | —           | `true` permits an unauthenticated non-loopback bind — honored only when running from a source checkout, never for a pip-installed package |
| `HKP_MAX_RUNTIMES_PER_USER` | — | Maximum runtimes one tenant may hold. Unset or `0` means unlimited. Re-creating a runtime that already exists is never refused. |
| `HKP_MAX_SERVICES_PER_RUNTIME` | — | Maximum services per runtime. Unset or `0` means unlimited. |
| `HKP_MIN_TIMER_INTERVAL_MS` | — | Lower bound on the Timer service's periodic interval; shorter periods are clamped. Unset or `0` means no floor. |
| `HKP_MAX_REQUEST_BODY_BYTES` | `26214400` | Largest request body accepted on a service endpoint (25 MB). Oversized requests get `413`. Set `0` to disable — unwise, since these endpoints take no token. |

Variables may also be placed in a `.env` file in the project root (real
environment variables win; set `SKIP_LOADING_ENV=1` to skip it).

Example:

```bash
PORT=9000 EXTERNAL_HOST=myhost.local python3 -m hkp
```

### Authentication

The auth model mirrors hkp-node exactly, so the same client (hkp-frontend, the
hkp-node coordinator) works against both runtimes:

- **Multi-tenant.** Runtimes are namespaced by the authenticated `sub` — the
  same identifier the hkp-node coordinator uses as its `userId`. Every route
  resolves runtime ids inside the caller's own namespace, so one instance serves
  many users: `GET /runtimes` lists only yours, `DELETE /runtimes` clears only
  yours, and a runtime id owned by someone else answers **404, not 403**, so ids
  cannot be probed. Two users may hold the same runtime id at once — the normal
  case, since boards ship stable ids (`node`, `chat-node`). Routes are unchanged:
  identity comes from the `Authorization` header, never the path. With auth off
  (loopback, or `ALLOW_NO_AUTH`), every request collapses into a single
  `anonymous` tenant — identical to the pre-tenancy behaviour.
- With `AUTH0_DOMAIN` + `AUTH0_AUDIENCE` set, every HTTP request needs
  `Authorization: Bearer <token>`; WebSocket upgrades take the token from the
  Authorization header or `?access_token=` (browsers can't set WS headers) and
  are Origin-checked against `ALLOWED_ORIGINS`.
- `POST /runtimes/{id}/session-token` (JWT-gated) mints an opaque in-memory
  token bound to the calling user and that runtime — used by the hkp-node
  coordinator for long-lived machine calls past JWT expiry. Tokens are purged
  when the runtime is removed.
- Fail closed: without Auth0 config the server only starts on a loopback bind
  (`HOST=127.0.0.1`), or from a source checkout with `ALLOW_NO_AUTH=true` — and
  in both cases it runs **without authentication** (every request is anonymous).
  Setting `ALLOWED_EMAILS` without Auth0 config refuses to start.
- Auth0 config always wins over the loopback bypass: with `AUTH0_DOMAIN` +
  `AUTH0_AUDIENCE` set, JWT auth is enforced on any bind — including
  `127.0.0.1`, which is how to test the authenticated path locally.

### Service endpoints (mounts)

Services that must be reachable from outside — currently `http-server-subservices`
— do not bind a port. The runtime assigns each one an opaque path on this same
server and publishes the resulting address in the service's state:

```
http://<EXTERNAL_HOST>:<PORT>/hosted/<mountId>
```

Ports are a single machine-wide namespace, so on a shared host a service asking
for a specific port is a land grab: the second claimant fails and whoever wins
receives traffic the other expected. An assigned id avoids that, and since
runtime ids are only unique per tenant they could not have appeared in a
globally-routable path anyway.

These endpoints are deliberately **unauthenticated** — they exist to be called by
outside parties (webhooks, uploads) that hold no token — so the unguessable
`mountId` is what gates access. It carries no user identifier. A mount is
released when its service is bypassed or its runtime goes away.

`port` is still accepted on the service and ignored, so existing boards load.

#### What a request looks like to the pipeline

A request reaches the pipeline as **MixedData** — JSON metadata plus the body —
matching hkp-node's service of the same name:

```jsonc
{
  "meta": {
    "method": "POST",
    "path": "/upload",              // path below the mount, not the mount prefix
    "query": { "a": "1" },
    "contentType": "application/json",  // when the request carried one
    "filename": "notes.txt"             // from content-disposition, when present
  },
  "body": { "hello": "world" }      // decoded, when the type allows it
}
```

The body arrives in **exactly one** form, never both:

| Content type                        | Field    | Value                |
| ----------------------------------- | -------- | -------------------- |
| `application/json`, `*+json`        | `body`   | parsed JSON value    |
| `application/x-www-form-urlencoded` | `body`   | parsed fields object |
| `text/*`                            | `body`   | string               |
| anything else                       | `binary` | raw bytes            |
| no request body (e.g. GET)          | —        | neither field        |

Charset parameters are ignored when matching. Malformed input falls back to
`binary` rather than failing the request.

**This replaced the previous flat `{ path, method }`.** A pipeline that matched
on `params.path` now needs `params.meta.path`.

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
| `speech-to-text`          | Transcribes mono `FloatRingBuffer` audio with a local Whisper model (requires the `asr` extra); `sampleRate` config declares the incoming rate (default 16000), other rates are resampled internally |
| `text-generation`         | Generates text with a local LLM — either via an OpenAI-compatible server (llama-server, Ollama, ...) or by loading a GGUF in-process (requires the `llm` extra); accepts a String prompt or JSON with `prompt`/`text`/`messages` |
| `text-to-speech`          | Synthesizes speech with the local Kokoro-82M model (requires the `tts` extra); accepts a String or JSON with `text`/`prompt`, emits a 24 kHz `FloatRingBuffer` |
| `skill-router`            | Matches free-form text against configured skills via the local LLM and emits `{ board, payload }` for dispatch (or `null` to stop) |

### Text generation backends

The `text-generation` service has two backends, selected by the `backend`
state (`"server"`, the default, or `"local"`).

**Server backend** — the service talks to any locally running server that
speaks the OpenAI chat-completions API (`POST {serverUrl}/v1/chat/completions`).
No extra Python dependencies are needed. This is the only way to run quants
that need a custom llama.cpp build, such as the reference model
**1-bit Bonsai 27B**
([prism-ml/Bonsai-27B-gguf](https://huggingface.co/prism-ml/Bonsai-27B-gguf)) —
a ~3.9 GB GGUF that retains ~90% of FP16 quality. Its `Q1_0_g128` format needs
the PrismML fork of llama.cpp (mainline llama.cpp and stock `llama-cpp-python`
cannot load it):

```bash
git clone https://github.com/PrismML-Eng/llama.cpp && cd llama.cpp
cmake -B build && cmake --build build -j        # Metal is the default on macOS
hf download prism-ml/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf --local-dir .
./build/bin/llama-server -m Bonsai-27B-Q1_0.gguf --port 8081 -ngl 99
```

**Local backend** — for standard GGUFs (Qwen, Llama, Mistral, ...) the service
loads the model in-process via `llama-cpp-python`, so no external server is
needed:

```bash
pip install -e ".[llm]"
```

Configure `backend: "local"` plus `modelPath` (path to the `.gguf` file;
`~` expands), and optionally `contextSize` (default 4096) and `gpuLayers`
(default -1 = offload all layers; Metal/CUDA support depends on how
`llama-cpp-python` was built). The model loads lazily on first use — status
notifications go `loading → generating → idle` — and reloads automatically
when `modelPath`, `contextSize`, or `gpuLayers` change.

The service's defaults (`serverUrl http://127.0.0.1:8081`, temperature 0.7,
top-p 0.95, top-k 20) match the Bonsai setup above and the model card's
recommended sampling parameters. For thinking models the reasoning is split
off into a separate `thinking` field of the output JSON; `text` carries only
the answer (the `thinking` on/off toggle itself is a llama-server extension
and applies to the server backend only).

Because the speech-to-text service emits JSON with a `text` key, the two chain
directly: `audio-input → speech-to-text → text-generation` turns voice notes
into LLM-processed text with no glue services.

### Text to speech

The `text-to-speech` service runs [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
in-process via [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)
(onnxruntime — no PyTorch, faster than real time on CPU):

```bash
pip install -e ".[tts]"
```

Model files (~310 MB onnx + ~27 MB voices, or ~88 MB with the
`kokoro-v1.0-int8` model) are downloaded automatically on first use into
`~/.cache/hkp-python/kokoro` (override with the `modelDir` state).

Config state: `model` (`kokoro-v1.0` | `kokoro-v1.0-int8`), `voice`
(default `af_heart`), `speed` (0.5–2.0), `lang` (e.g. `en-us`), `modelDir`.
Input is a String or JSON with a `text`/`prompt` key — the text-generation
service's output pipes straight in. Output is a 24 kHz mono `FloatRingBuffer`,
which the browser Audio Output service plays directly (set its `sampleRate`
state to 24000).

The three services complete a fully local voice loop with no glue:
`audio-input → speech-to-text → text-generation → text-to-speech → audio-output`.

### Skill routing

The `skill-router` service turns free-form text (e.g. a voice transcript) into
a dispatch decision. It shares the LLM backends of `text-generation` (one LLM
call, strict JSON output, thinking disabled): `backend: "server"` (default)
talks to an OpenAI-compatible server via `serverUrl`, `backend: "local"` loads
a GGUF in-process via `modelPath`/`contextSize`/`gpuLayers` and the `[llm]`
extra. Configure it with a `skills` array:

```json
{
  "skills": [
    {
      "action": "send notification",
      "board": "send ntfy",
      "payload": { "topic": "the ntfy topic", "message": "the message text" }
    }
  ]
}
```

The payload template's keys are the parameter names; its values describe what
the model should extract. On a match the service emits
`{ "board": "send ntfy", "payload": { "topic": "X", "message": "hello" } }` —
in the browser, a Board-Service in "board name from input" mode plays that
saved board with the payload. Backend errors surface as `{ "error": ... }`.

A match is **early-returned** (`ControlFlowData`): services after the router in
the same runtime are skipped and the dispatch payload becomes the runtime
output. The `noMatch` config decides the other branch: `"stop"` (default) ends
the pipeline, `"forward"` passes the original input to the remaining services —
chain `skill-router (noMatch: forward) → text-generation → text-to-speech` for
a voice assistant that acts on matching requests and answers everything else.
See `hkp-frontend/boards/skill-router-demo-board.json`,
`voice-assistant-skills-demo-board.json`, and the target board
`send-ntfy-board.json`.

## Testing

```bash
./run_tests.sh        # runs all tests
./run_tests.sh -v     # verbose
./run_tests.sh -k map # filter by name
```

The script creates `.venv` automatically if it does not exist.
