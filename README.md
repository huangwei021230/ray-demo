# ray-demo

Two complementary demos for learning Ray:

1. **Simulator** (`main.py` + `ray_sim/` + `static/`) — a step-by-step visualization of Ray's paper architecture (GCS, schedulers, object store, execution engine), driven from a vanilla-JS/SVG frontend over WebSocket.
2. **Real Ray demo** (`ray_real_demo.py`) — a tiny distributed SGD job on a real local Ray cluster, designed to be slow enough to browse the dashboard while it runs.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync
```

This reads `pyproject.toml` + `uv.lock` and creates `.venv/` with pinned dependencies.

To use the existing `venv-ray/` directory instead:

```bash
UV_PROJECT_ENVIRONMENT=venv-ray uv sync
```

## Run the simulator

```bash
./.venv/bin/python main.py
```

FastAPI server on `0.0.0.0:8000`. Open http://localhost:8000 in a browser.

## Run the real Ray demo

One-time setup — Ray's plasma socket path has a 107-byte cap, and this project path is too long, so we route the temp dir through a short symlink:

```bash
mkdir -p ray_session
ln -sfn "$(pwd)/ray_session" ~/.rd
```

Then:

```bash
./.venv/bin/python -u ray_real_demo.py
```

- Dashboard: http://127.0.0.1:8266 — browse **Jobs / Actors / Tasks / Cluster** tabs while it trains
- Trains 200 SGD steps fitting `y = 2x + 1` across 4 worker actors + 1 reducer task
- Stays alive after training so the dashboard remains reachable; Ctrl-C to exit

## Layout

- `main.py` — simulator entrypoint, calls `server.run_server()`
- `server.py` — FastAPI app with `/ws/{session_id}` WebSocket, serves `static/`
- `ray_sim/` — simulation engine (GCS, schedulers, object store, execution)
- `static/` — single-page frontend
- `ray_real_demo.py` — real distributed SGD on a local Ray cluster
- `pyproject.toml` / `uv.lock` / `.python-version` — reproducible env via uv
