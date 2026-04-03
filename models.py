from pydantic import Field, ConfigDict
from typing import List, Optional
from enum import Enum
from openenv.core.env_server.types import Action, Observation, State     


class ActionType(str, Enum):
    ASSIGN_DOCTOR = "assign_doctor"
    WAIT = "wait"


class HospitalTriageAction(Action):
    action_type: ActionType = Field(..., description="Type of action")
    patient_id: Optional[int] = Field(None, description="Patient ID")
    doctor_id: Optional[int] = Field(None, description="Doctor ID")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "action_type": "assign_doctor",
                "patient_id": 0,
                "doctor_id": 0
            }
        }
    )


class HospitalTriageObservation(Observation):
    waiting_patients: List[int] = Field(default_factory=list)
    free_doctors: List[int] = Field(default_factory=list)
    critical_patients: List[int] = Field(default_factory=list)
    waiting_count: int = 0
    critical_count: int = 0
    free_doctor_count: int = 0
    step_ratio: float = 0.0
    treated_ratio: float = 0.0
    
    model_config = ConfigDict(extra="allow")


class HospitalTriageState(State):
    total_patients: int = 0
    treated_patients: int = 0
    critical_treated: int = 0
    total_critical: int = 0
    efficiency: float = 0.0
    current_task: str = "easy_triage"
    episode_reward: float = 0.0