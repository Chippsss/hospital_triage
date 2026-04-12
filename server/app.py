import os
from functools import partial
from fastapi import FastAPI
from openenv.core.env_server import create_web_interface_app

from server.hospital_triage_environment import HospitalTriageEnvironment
from models import HospitalTriageAction, HospitalTriageObservation

# Get task from environment variable (default to easy_triage)
_task = os.getenv("TASK_NAME", "easy_triage")

def create_task_env():
    return HospitalTriageEnvironment(task_id=_task)

app = create_web_interface_app(
    env=create_task_env,
    action_cls=HospitalTriageAction,
    observation_cls=HospitalTriageObservation,
    env_name=_task,
)

@app.get("/health")
def health():
    return {"status": "ok", "environment": "hospital_triage", "task": _task}

def main():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()