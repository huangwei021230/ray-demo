# AGENTS.md

## Run

```bash
python main.py
```

Starts the FastAPI server on `0.0.0.0:8000`. No virtual env config, no lock files — install dependencies ad-hoc (`fastapi`, `uvicorn`).

## Architecture

- `main.py` — entrypoint, just calls `server.run_server()`
- `server.py` — FastAPI app with WebSocket at `/ws/{session_id}`, serves static files from `static/`
- `ray_sim/` — core simulation engine (GCS, schedulers, object store, execution engine)
- `static/` — single-page frontend (vanilla JS + SVG, no framework)

The engine simulates Ray's paper architecture step-by-step. Programs are defined as lists of `ProgramOp` dataclasses in `ray_sim/programs.py`. The `ExecutionEngine.run_program()` method executes them and records a `SystemSnapshot` per step, sent to the frontend via WebSocket.

## WebSocket protocol

Client sends JSON commands: `{action: "load", program: "add"}`, `{action: "step"}`, `{action: "back"}`, `{action: "goto", step: N}`, `{action: "autoplay", speed: ms}`, `{action: "reset"}`.

Server responds with `{type: "state", ...}` containing the full system snapshot.

## Module import path

`main.py` injects the project root into `sys.path` so imports like `from ray_sim import ...` work. When running tests or scripts that import `ray_sim`, ensure the project root is on `sys.path`.
