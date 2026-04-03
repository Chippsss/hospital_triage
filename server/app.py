# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Hospital Triage Environment.
"""

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


def main():
    """Entry point for the server."""
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()