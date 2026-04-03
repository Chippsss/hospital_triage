import requests
import json

# Test health
response = requests.get("http://localhost:8000/health")
print(f"Health: {response.json()}")

# Test reset
reset_data = {"episode_id": "test"}
response = requests.post("http://localhost:8000/reset", json=reset_data)
print(f"Reset response: {json.dumps(response.json(), indent=2)}")