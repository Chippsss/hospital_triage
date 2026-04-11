# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
FastAPI application for the Hospital Triage Environment.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import HospitalTriageAction, HospitalTriageObservation
    from .hospital_triage_environment import HospitalTriageEnvironment
except ImportError:
    from models import HospitalTriageAction, HospitalTriageObservation
    from server.hospital_triage_environment import HospitalTriageEnvironment


# Create the app with web interface
app = create_app(
    HospitalTriageEnvironment,
    HospitalTriageAction,
    HospitalTriageObservation,
    env_name="hospital_triage",
    max_concurrent_envs=10,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add a simple health check at root
@app.get("/")
async def root():
    return {"status": "healthy", "message": "Hospital Triage Environment is running"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)