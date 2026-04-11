# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
FastAPI application for the Hospital Triage Environment.
"""

import os
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


# Create the app with web interface and README integration
app = create_app(
    HospitalTriageEnvironment,
    HospitalTriageAction,
    HospitalTriageObservation,
    env_name="hospital_triage",
    max_concurrent_envs=10,
)


if __name__ == "__main__":
    import uvicorn
    # HF Spaces REQUIRES port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)