"""
Ray Demo - FastAPI Server with WebSocket

Serves the frontend and provides WebSocket-based step-by-step control
for the Ray simulation engine.
"""

import json
from typing import Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from ray_sim import ExecutionEngine, ALL_PROGRAMS

app = FastAPI(title="Ray Paper Demo")

# Path to static files
STATIC_DIR = Path(__file__).parent / "static"

# Active sessions
sessions: Dict[str, ExecutionEngine] = {}


@app.get("/")
async def index():
    """Serve the main page."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/programs")
async def list_programs():
    """List available example programs."""
    programs = []
    for key, factory in ALL_PROGRAMS.items():
        prog = factory()
        programs.append({
            "id": key,
            "name": prog.name,
            "description": prog.description,
            "paper_mapping": prog.paper_mapping,
            "num_nodes": prog.num_nodes,
        })
    return programs


@app.get("/api/programs/{program_id}")
async def get_program(program_id: str):
    """Get details of a specific program."""
    if program_id not in ALL_PROGRAMS:
        return {"error": "program not found"}
    prog = ALL_PROGRAMS[program_id]()
    return {
        "id": program_id,
        "name": prog.name,
        "description": prog.description,
        "paper_mapping": prog.paper_mapping,
        "num_nodes": prog.num_nodes,
        "num_operations": len(prog.operations),
    }


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for step-by-step execution control.
    
    Protocol:
    - Client sends JSON commands: {action: "...", ...}
    - Server sends JSON state updates: {type: "state", ...}
    
    Commands:
    - {action: "load", program: "add"}  — Load a program
    - {action: "step"}                   — Step forward
    - {action: "back"}                   — Step backward
    - {action: "goto", step: N}          — Go to step N
    - {action: "reset"}                  — Reset current program
    """
    await websocket.accept()
    
    engine: Optional[ExecutionEngine] = None
    current_step = 0
    
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            action = msg.get("action", "")
            
            if action == "load":
                # Load and execute a program
                program_id = msg.get("program", "add")
                if program_id not in ALL_PROGRAMS:
                    await websocket.send_json({"type": "error", "message": f"Unknown program: {program_id}"})
                    continue
                
                program = ALL_PROGRAMS[program_id]()
                engine = ExecutionEngine()
                engine.run_program(program)
                sessions[session_id] = engine
                current_step = 0
                
                # Send program info first
                await websocket.send_json({
                    "type": "program_loaded",
                    "name": program.name,
                    "description": program.description,
                    "paper_mapping": program.paper_mapping,
                    "num_nodes": program.num_nodes,
                    "total_steps": engine.get_total_steps(),
                })
                
                # Then send initial state
                state = engine.to_json(current_step)
                state["type"] = "state"
                await websocket.send_json(state)
            
            elif action == "step":
                if engine is None:
                    await websocket.send_json({"type": "error", "message": "No program loaded"})
                    continue
                
                if current_step < engine.get_total_steps() - 1:
                    current_step += 1
                    state = engine.to_json(current_step)
                    state["type"] = "state"
                    await websocket.send_json(state)
                else:
                    await websocket.send_json({"type": "info", "message": "Already at the last step"})
            
            elif action == "back":
                if engine is None:
                    await websocket.send_json({"type": "error", "message": "No program loaded"})
                    continue
                
                if current_step > 0:
                    current_step -= 1
                    state = engine.to_json(current_step)
                    state["type"] = "state"
                    await websocket.send_json(state)
                else:
                    await websocket.send_json({"type": "info", "message": "Already at the first step"})
            
            elif action == "goto":
                if engine is None:
                    await websocket.send_json({"type": "error", "message": "No program loaded"})
                    continue
                
                target_step = msg.get("step", 0)
                target_step = max(0, min(target_step, engine.get_total_steps() - 1))
                current_step = target_step
                state = engine.to_json(current_step)
                state["type"] = "state"
                await websocket.send_json(state)
            
            elif action == "reset":
                # Rewind to step 0 of the currently loaded program (do not
                # destroy the engine — the user wants to replay, not unload).
                if engine is None:
                    await websocket.send_json({"type": "error", "message": "No program loaded"})
                    continue
                current_step = 0
                state = engine.to_json(current_step)
                state["type"] = "state"
                await websocket.send_json(state)
            
            else:
                await websocket.send_json({"type": "error", "message": f"Unknown action: {action}"})
    
    except WebSocketDisconnect:
        if session_id in sessions:
            del sessions[session_id]


# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the development server."""
    import uvicorn
    print(f"\n{'='*60}")
    print(f"  Ray Paper Demo - Interactive Visualization")
    print(f"  Open http://localhost:{port} in your browser")
    print(f"{'='*60}\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
