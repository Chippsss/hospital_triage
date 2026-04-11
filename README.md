---
title: Hospital Triage Environment
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "1.0"
app_file: server/app.py
pinned: false
port: 7860
---

# Hospital Triage Environment

A realistic hospital resource allocation environment for OpenEnv where an AI agent learns to manage doctors and prioritize patients in an emergency department.

## Environment Description

Hospital emergency departments face constant challenges: limited doctors, varying patient criticality, and time-sensitive decisions. This environment simulates these real-world constraints, requiring agents to:

- Allocate limited doctor resources efficiently
- Prioritize critical patients appropriately  
- Minimize patient wait times
- Maximize overall treatment throughput

The environment simulates a hospital with:
- **2-5 doctors** with varying efficiency levels
- **5-20 patients** with different criticality levels (30-60% critical)
- **20-30 steps** per episode (time-limited)

## Action Space

Three action types:

| Action | Description | Parameters |
|--------|-------------|------------|
| `assign_doctor` | Assign a doctor to treat a patient | `patient_id`, `doctor_id` |
| `wait` | Do nothing, advance time | None |
| `prioritize` | Increase a patient's priority | `patient_id` |

### Action Format

```python
# Assign doctor to patient
HospitalTriageAction(
    action_type="assign_doctor",
    patient_id=0,
    doctor_id=0
)
```

# Wait
HospitalTriageAction(action_type="wait")

text
[OK] hospital_triage: Ready for multi-mode deployment


## Observation Space

The agent observes the following state:

| Field | Type | Description |
|-------|------|-------------|
| `waiting_patients` | List[int] | IDs of untreated patients |
| `free_doctors` | List[int] | IDs of available doctors |
| `critical_patients` | List[int] | IDs of critical untreated patients |
| `waiting_count` | int | Number of waiting patients |
| `critical_count` | int | Number of critical patients |
| `free_doctor_count` | int | Number of free doctors |
| `step_ratio` | float | Progress through episode (0 to 1) |
| `treated_ratio` | float | Proportion of patients treated |


## Tasks

The environment includes 3 tasks with increasing difficulty:

### Easy: Basic Triage
- **Doctors**: 2-3
- **Patients**: 5-7
- **Critical ratio**: 30%
- **Max steps**: 20
- **Expected score**: 0.85-0.95

### Medium: Medium Hospital Management
- **Doctors**: 3-4
- **Patients**: 8-12
- **Critical ratio**: 40%
- **Max steps**: 25
- **Expected score**: 0.75-0.85

### Hard: Emergency Overload
- **Doctors**: 4-5
- **Patients**: 15-20
- **Critical ratio**: 60%
- **Max steps**: 30
- **Expected score**: 0.65-0.75

## Reward Structure

| Action | Reward |
|--------|--------|
| Treat normal patient | +1.0 |
| Treat critical patient | +3.0 (bonus) |
| Invalid action | -0.3 to -0.5 |
| Per step penalty | -0.02 |
| Patient wait time penalty | -0.05 to -0.15 |
| Early completion bonus | Up to +2.0 |

## Setup Instructions

### Prerequisites
- Python 3.11 or higher
- Docker (optional, for containerized deployment)
- OpenAI API key (for inference)

### Local Installation

```bash
# Clone the repository
git clone https://github.com/Chippsss/hospital_triage.git
cd hospital_triage

# Install dependencies
pip install -e .

# Start the server
uvicorn server.app:app --reload --port 8000
```

# Build the image
docker build -t hospital_triage .

# Run the container
docker run -p 8000:8000 hospital_triage

# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Run inference
python inference.py

## Baseline Scores

Using heuristic policy (prioritize critical patients):

| Task | Score | Steps | Success Rate |
|------|-------|-------|--------------|
| Basic Triage | 0.92 | 7 | 100% |
| Medium Hospital | 0.87 | 12 | 100% |
| Emergency Overload | 0.81 | 20 | 100% |
| **Average** | **0.87** | - | - |

## Live Environment

The environment is deployed on Hugging Face Spaces:
- **Space URL**: https://chinuT-hospital_triage.hf.space
- **API Endpoint**: https://chinuT-hospital_triage.hf.space
- **Web Interface**: https://chinuT-hospital_triage.hf.space/web

- ## Usage Examples

### Connect from Python

```python
from hospital_triage.client import HospitalTriageEnv
from hospital_triage.models import HospitalTriageAction, ActionType

# Connect to local or remote environment
async with HospitalTriageEnv(base_url="http://localhost:8000") as env:
    # Reset with specific task
    result = await env.reset(task_id="easy_triage")
    
    # Take an action
    action = HospitalTriageAction(
        action_type=ActionType.ASSIGN_DOCTOR,
        patient_id=0,
        doctor_id=0
    )
    result = await env.step(action)
    
    print(f"Reward: {result.reward}")
    print(f"Waiting patients: {result.observation.waiting_patients}")
```

# Health check
curl https://chinuT-hospital_triage.hf.space/health

# Reset environment
curl -X POST https://chinuT-hospital_triage.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"episode_id": "test", "task_id": "easy_triage"}'

  ## Validation

Run the OpenEnv validation:

```bash
openenv validate
```

# Expected output:
[OK] hospital_triage: Ready for multi-mode deployment
