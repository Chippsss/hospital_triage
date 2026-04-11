# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
FastAPI application for the Hospital Triage Environment.
"""

import os
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from ..models import HospitalTriageAction, HospitalTriageObservation
    from .hospital_triage_environment import HospitalTriageEnvironment
except ImportError:
    from models import HospitalTriageAction, HospitalTriageObservation
    from server.hospital_triage_environment import HospitalTriageEnvironment


# Create FastAPI app directly
app = FastAPI(title="Hospital Triage Environment", description="Hospital triage simulation for RL agents")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
_env = None

def get_env():
    global _env
    if _env is None:
        _env = HospitalTriageEnvironment()
    return _env


class ResetRequest(BaseModel):
    episode_id: str = None
    seed: int = None
    task_id: str = None


class StepRequest(BaseModel):
    episode_id: str = None
    action: HospitalTriageAction


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/reset")
async def reset(request: ResetRequest):
    env = get_env()
    obs = env.reset(seed=request.seed, task_id=request.task_id)
    return {"observation": obs.model_dump(), "reward": None, "done": False}


@app.post("/step")
async def step(request: StepRequest):
    env = get_env()
    obs = env.step(request.action)
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}


@app.get("/state")
async def state():
    env = get_env()
    return env.state.model_dump()


@app.get("/web")
async def web():
    from fastapi.responses import HTMLResponse
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hospital Triage Environment</title>
        <style>
            body { font-family: Arial; padding: 20px; }
            button { margin: 5px; padding: 10px; }
            pre { background: #f0f0f0; padding: 10px; }
        </style>
    </head>
    <body>
        <h1>🏥 Hospital Triage Environment</h1>
        <button onclick="reset()">Reset</button>
        <button onclick="step()">Step</button>
        <button onclick="getState()">Get State</button>
        <h3>Response:</h3>
        <pre id="output">{}</pre>
        <script>
            async function reset() {
                const res = await fetch('/reset', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
                const data = await res.json();
                document.getElementById('output').innerText = JSON.stringify(data, null, 2);
            }
            async function step() {
                const action = {"action_type": "assign_doctor", "patient_id": 0, "doctor_id": 0};
                const res = await fetch('/step', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action: action})});
                const data = await res.json();
                document.getElementById('output').innerText = JSON.stringify(data, null, 2);
            }
            async function getState() {
                const res = await fetch('/state');
                const data = await res.json();
                document.getElementById('output').innerText = JSON.stringify(data, null, 2);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)