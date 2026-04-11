FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir openenv-core fastapi uvicorn pydantic openai

# Copy your code
COPY . .

# Expose port 7860 (HF Spaces requirement)
EXPOSE 7860

# Run on port 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]