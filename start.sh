#!/bin/bash
# Start script for Hugging Face Space

# Set environment variables
export TASK_NAME=${TASK_NAME:-easy_triage}
export PORT=${PORT:-8000}

# Run the FastAPI server
cd /app
uvicorn server.app:app --host 0.0.0.0 --port $PORT