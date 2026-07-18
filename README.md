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
  (`HOST=127.0.0.1`), or from a source checkout with `ALLOW_NO_AUTH=true` — and
  in both cases it runs **without authentication** (every request is anonymous).
  Setting `ALLOWED_EMAILS` without Auth0 config refuses to start.
- Auth0 config always wins over the loopback bypass: with `AUTH0_DOMAIN` +
  `AUTH0_AUDIENCE` set, JWT auth is enforced on any bind — including
  `127.0.0.1`, which is how to test the authenticated path locally.

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
| `text-generation`         | Generates text via a local OpenAI-compatible server (llama-server, Ollama, ...); accepts a String prompt or JSON with `prompt`/`text`/`messages` |
| `text-to-speech`          | Synthesizes speech with the local Kokoro-82M model (requires the `tts` extra); accepts a String or JSON with `text`/`prompt`, emits a 24 kHz `FloatRingBuffer` |
| `skill-router`            | Matches free-form text against configured skills via the local LLM and emits `{ board, payload }` for dispatch (or `null` to stop) |

### Text generation backend

The `text-generation` service does not run a model in-process — it talks to any
locally running server that speaks the OpenAI chat-completions API
(`POST {serverUrl}/v1/chat/completions`). No extra Python dependencies are needed.

The reference backend is **1-bit Bonsai 27B**
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

The service's defaults (`serverUrl http://127.0.0.1:8081`, temperature 0.7,
top-p 0.95, top-k 20) match this setup and the Bonsai model card's recommended
sampling parameters. For thinking models the reasoning is split off into a
separate `thinking` field of the output JSON; `text` carries only the answer.

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
a dispatch decision. It shares the OpenAI-compatible backend of
`text-generation` (one LLM call, strict JSON output, thinking disabled).
Configure it with a `skills` array:

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
