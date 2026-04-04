# hkp-python

asyncio port of the hkp-node runtime engine, built with [aiohttp](https://docs.aiohttp.org/).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
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
| `ALLOWED_ORIGINS` | `*`         | CORS allowed origins                                          |

Example:

```bash
PORT=9000 EXTERNAL_HOST=myhost.local python3 -m hkp
```

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

## Testing

```bash
./run_tests.sh        # runs all tests
./run_tests.sh -v     # verbose
./run_tests.sh -k map # filter by name
```

The script creates `.venv` automatically if it does not exist.
