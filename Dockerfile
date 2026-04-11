FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install openenv-core fastapi uvicorn pydantic

# Copy your code
COPY . .

# Run the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]