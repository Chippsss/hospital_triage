# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hospital Triage Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import HospitalTriageAction, HospitalTriageObservation, HospitalTriageState
except ImportError:
    from models import HospitalTriageAction, HospitalTriageObservation, HospitalTriageState

class HospitalTriageEnv(
    EnvClient[HospitalTriageAction, HospitalTriageObservation, HospitalTriageState]
):
    """
    Client for the Hospital Triage Environment.
    """

    def _step_payload(self, action: HospitalTriageAction) -> Dict:
        """
        Convert HospitalTriageAction to JSON payload for step message.
        """
        return {
            "action_type": action.action_type.value,
            "patient_id": action.patient_id,
            "doctor_id": action.doctor_id,
        }

    def _parse_result(self, payload: Dict) -> StepResult[HospitalTriageObservation]:
        """
        Parse server response into StepResult.
        """
        obs_data = payload.get("observation", {})
        
        observation = HospitalTriageObservation(
            waiting_patients=obs_data.get("waiting_patients", []),
            free_doctors=obs_data.get("free_doctors", []),
            critical_patients=obs_data.get("critical_patients", []),
            waiting_count=obs_data.get("waiting_count", 0),
            critical_count=obs_data.get("critical_count", 0),
            free_doctor_count=obs_data.get("free_doctor_count", 0),
            step_ratio=obs_data.get("step_ratio", 0.0),
            treated_ratio=obs_data.get("treated_ratio", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> HospitalTriageState:
        """
        Parse server response into State object.
        """
        return HospitalTriageState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            total_patients=payload.get("total_patients", 0),
            treated_patients=payload.get("treated_patients", 0),
            critical_treated=payload.get("critical_treated", 0),
            total_critical=payload.get("total_critical", 0),
            efficiency=payload.get("efficiency", 0.0),
            current_task=payload.get("current_task", "easy_triage"),
            episode_reward=payload.get("episode_reward", 0.0),
        )