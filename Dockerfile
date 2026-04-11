FROM python:3.11-slim

WORKDIR /app

# Copy requirements.txt (it now exists)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire code
COPY . .

# Run your server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]