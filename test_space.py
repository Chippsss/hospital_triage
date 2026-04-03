import requests

# These are the correct API endpoints (not huggingface.co domain)
urls = [
    "https://chinuT-hospital_triage.hf.space/health",
    "https://chinuT-hospital_triage.hf.space/docs", 
    "https://chinuT-hospital_triage.hf.space/web",
]

for url in urls:
    try:
        response = requests.get(url, timeout=10)
        print(f"{url}: {response.status_code}")
        if response.status_code == 200:
            print(f"  ✓ Working!")
            if "/health" in url:
                print(f"  Response: {response.json()}")
    except Exception as e:
        print(f"{url}: Error - {e}")