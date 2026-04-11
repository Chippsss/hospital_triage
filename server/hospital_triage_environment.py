# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Hospital Triage Environment Implementation with nuanced scoring.
"""

import random
from uuid import uuid4
from typing import Optional, Tuple, Dict, Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import HospitalTriageAction, HospitalTriageObservation, HospitalTriageState, ActionType
except ImportError:
    from models import HospitalTriageAction, HospitalTriageObservation, HospitalTriageState, ActionType


class HospitalTriageEnvironment(Environment):
    """
    Hospital triage environment for training AI agents.
    
    The agent controls doctor assignments to treat patients efficiently.
    Critical patients give higher rewards and should be prioritized.
    """
    
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    
    # Task configurations with realistic difficulty
    TASKS = {
        "easy_triage": {
            "name": "Basic Triage",
            "max_steps": 20,
            "doctors_range": (2, 3),
            "patients_range": (5, 7),
            "critical_ratio": 0.3,
            "reward_multiplier": 1.0,
            "optimal_score": 0.95  # Even optimal play can't get 1.0
        },
        "medium_triage": {
            "name": "Medium Hospital Management",
            "max_steps": 25,
            "doctors_range": (3, 4),
            "patients_range": (8, 12),
            "critical_ratio": 0.4,
            "reward_multiplier": 1.5,
            "optimal_score": 0.90
        },
        "hard_triage": {
            "name": "Emergency Overload",
            "max_steps": 30,
            "doctors_range": (4, 5),
            "patients_range": (15, 20),
            "critical_ratio": 0.6,
            "reward_multiplier": 2.0,
            "optimal_score": 0.85
        }
    }

    def __init__(self):
        """Initialize the hospital triage environment."""
        self._state = HospitalTriageState(
            episode_id=str(uuid4()),
            step_count=0,
            total_patients=0,
            treated_patients=0,
            critical_treated=0,
            total_critical=0,
            efficiency=0.0,
            current_task="easy_triage",
            episode_reward=0.0
        )
        self._current_task = "easy_triage"
        self._doctors = []
        self._patients = []
        self._step_count = 0
        self._max_steps = 20
        self._total_reward = 0.0
        self._seed = None

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, 
              task_id: Optional[str] = None) -> HospitalTriageObservation:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed for reproducibility (makes scores vary)
            episode_id: Unique identifier for this episode
            task_id: Which task to run (easy_triage, medium_triage, hard_triage)
        
        Returns:
            Initial observation
        """
        # Use random seed if not provided (ensures score variance)
        if seed is None:
            seed = random.randint(0, 100000)
        self._seed = seed
        random.seed(seed)
        
        # Set task
        if task_id and task_id in self.TASKS:
            self._current_task = task_id
        else:
            self._current_task = "easy_triage"
        
        task_config = self.TASKS[self._current_task]
        
        # Reset episode counters
        self._step_count = 0
        self._max_steps = task_config["max_steps"]
        self._total_reward = 0.0
        
        # Create doctors with varying efficiency
        min_doctors, max_doctors = task_config["doctors_range"]
        num_doctors = random.randint(min_doctors, max_doctors)
        self._doctors = [
            {
                "id": i,
                "busy": False,
                "treatments": 0,
                "efficiency": random.uniform(0.7, 1.3),  # Varying doctor skill
                "exhaustion": 0.0
            }
            for i in range(num_doctors)
        ]
        
        # Create patients with varying criticality and arrival times
        min_patients, max_patients = task_config["patients_range"]
        num_patients = random.randint(min_patients, max_patients)
        self._patients = []
        critical_count = 0
        
        for i in range(num_patients):
            is_critical = random.random() < task_config["critical_ratio"]
            if is_critical:
                critical_count += 1
            
            self._patients.append({
                "id": i,
                "critical": is_critical,
                "treated": False,
                "wait_time": 0,
                "priority": 2 if is_critical else 1,
                "arrival_step": random.randint(0, 5),  # Staggered arrival
                "complexity": random.uniform(0.8, 1.2)  # Varying treatment complexity
            })
        
        # Sort by arrival step (some patients arrive later)
        self._patients.sort(key=lambda x: x["arrival_step"])
        
        # Initialize state
        episode_id = episode_id or str(uuid4())
        self._state = HospitalTriageState(
            episode_id=episode_id,
            step_count=0,
            total_patients=num_patients,
            treated_patients=0,
            critical_treated=0,
            total_critical=critical_count,
            efficiency=0.0,
            current_task=self._current_task,
            episode_reward=0.0
        )
        
        return self._get_observation()

    def step(self, action: HospitalTriageAction) -> HospitalTriageObservation:
        """
        Execute a step in the environment.
        
        Args:
            action: HospitalTriageAction (assign_doctor or wait)
        
        Returns:
            Observation with updated hospital state
        """
        self._step_count += 1
        reward = 0.0
        
        # Process the action
        if action.action_type == ActionType.ASSIGN_DOCTOR:
            reward += self._process_assignment(action)
        else:  # WAIT action
            reward -= 0.08  # Higher wait penalty
        
        # Time penalty to encourage efficiency
        reward -= 0.03
        
        # Update wait times and apply penalties
        for patient in self._patients:
            if not patient["treated"] and patient["arrival_step"] <= self._step_count:
                patient["wait_time"] += 1
                # Progressive waiting penalties
                if patient["critical"]:
                    if patient["wait_time"] > 3:
                        reward -= 0.15
                    elif patient["wait_time"] > 5:
                        reward -= 0.25
                else:
                    if patient["wait_time"] > 5:
                        reward -= 0.08
                    elif patient["wait_time"] > 8:
                        reward -= 0.12
        
        # Doctor exhaustion penalty
        for doctor in self._doctors:
            if doctor["treatments"] > 5:
                doctor["exhaustion"] = min(0.3, doctor["exhaustion"] + 0.02)
                reward -= doctor["exhaustion"] * 0.1
        
        # Check if episode is done
        available_patients = [p for p in self._patients if not p["treated"] and p["arrival_step"] <= self._step_count]
        all_treated = len(available_patients) == 0 and self._step_count > max([p["arrival_step"] for p in self._patients])
        done = all_treated or self._step_count >= self._max_steps
        
        # Early completion bonus (scaled to prevent perfect scores)
        if all_treated and self._step_count < self._max_steps:
            early_bonus = ((self._max_steps - self._step_count) / self._max_steps) * 1.5
            reward += early_bonus * 0.8  # Reduced bonus
        
        # Update total reward and state
        self._total_reward += reward
        self._state.episode_reward = self._total_reward
        self._state.step_count = self._step_count
        self._state.treated_patients = sum(1 for p in self._patients if p["treated"])
        
        observation = self._get_observation()
        observation.done = done
        observation.reward = reward if not done else None
        
        return observation

    def _process_assignment(self, action: HospitalTriageAction) -> float:
        """Process a doctor-patient assignment with nuanced rewards."""
        reward = 0.0
        
        # Validate action
        if (action.patient_id is None or action.doctor_id is None or
            action.patient_id >= len(self._patients) or
            action.doctor_id >= len(self._doctors)):
            return -0.3
        
        patient = self._patients[action.patient_id]
        doctor = self._doctors[action.doctor_id]
        
        # Check if patient has arrived yet
        if patient["arrival_step"] > self._step_count:
            return -0.2
        
        # Check if assignment is valid
        if doctor["busy"] or patient["treated"]:
            return -0.35
        
        # Process treatment
        doctor["busy"] = True
        patient["treated"] = True
        doctor["treatments"] += 1
        
        # Base reward (reduced to prevent perfect accumulation)
        reward += 0.8
        
        # Critical patient bonus (reduced)
        if patient["critical"]:
            reward += 1.5
            self._state.critical_treated += 1
        
        # Doctor efficiency multiplier
        reward *= max(0.5, min(1.5, doctor["efficiency"]))
        
        # Wait time penalty (more severe)
        wait_penalty = min(patient["wait_time"] * 0.08, 0.6)
        reward -= wait_penalty
        
        # Complexity modifier
        reward *= patient["complexity"]
        
        # Exhaustion penalty
        reward *= (1.0 - doctor["exhaustion"])
        
        # Free the doctor
        doctor["busy"] = False
        
        # Apply task multiplier
        task_config = self.TASKS[self._current_task]
        reward *= task_config["reward_multiplier"]
        
        return max(-0.5, min(2.5, reward))  # Clamp rewards

    def _get_observation(self) -> HospitalTriageObservation:
        """Build the current observation."""
        # Only show patients that have arrived
        available_patients = [p for p in self._patients if not p["treated"] and p["arrival_step"] <= self._step_count]
        waiting = [p["id"] for p in available_patients]
        free_doctors = [d["id"] for d in self._doctors if not d["busy"]]
        critical = [p["id"] for p in available_patients if p["critical"]]
        
        treated = sum(1 for p in self._patients if p["treated"])
        total_arrived = len([p for p in self._patients if p["arrival_step"] <= self._step_count])
        treated_ratio = treated / total_arrived if total_arrived > 0 else 0
        
        return HospitalTriageObservation(
            waiting_patients=waiting,
            free_doctors=free_doctors,
            critical_patients=critical,
            waiting_count=len(waiting),
            critical_count=len(critical),
            free_doctor_count=len(free_doctors),
            step_ratio=self._step_count / self._max_steps if self._max_steps > 0 else 0,
            treated_ratio=treated_ratio,
            done=False,
            reward=None
        )

    @property
    def state(self) -> HospitalTriageState:
        """Return the current state."""
        self._state.efficiency = (
            self._state.treated_patients / self._step_count
            if self._step_count > 0 else 0
        )
        return self._state

    def _calculate_score(self) -> float:
        """
        Calculate final score for the episode.
        IMPORTANT: Score must be strictly between 0 and 1 (never 0.0 or 1.0)
        """
        if not self._patients:
            return 0.500  # Return 0.5, not 0.85

        treated = sum(1 for p in self._patients if p["treated"])
        total_arrived = len([p for p in self._patients if p["arrival_step"] <= self._max_steps])

        if total_arrived == 0:
            return 0.500

        # Calculate scores
        treatment_score = treated / total_arrived
        critical_score = (
            self._state.critical_treated / self._state.total_critical
            if self._state.total_critical > 0 else 0.9
        )

        # Efficiency score
        theoretical_max = total_arrived / 2
        efficiency = (treated / max(self._step_count, 1)) / theoretical_max if theoretical_max > 0 else 0
        efficiency_score = min(efficiency, 0.95)

        # Utilization score
        total_treatments = sum(d["treatments"] for d in self._doctors)
        utilization = total_treatments / (self._step_count * len(self._doctors)) if self._step_count > 0 else 0
        utilization_score = min(utilization, 0.9)

        # Wait penalty
        avg_wait_time = sum(p["wait_time"] for p in self._patients if p["treated"]) / max(treated, 1)
        wait_penalty = min(avg_wait_time / 10, 0.2)

        # Combine
        score = (
            treatment_score * 0.35 +
            critical_score * 0.30 +
            efficiency_score * 0.20 +
            utilization_score * 0.15 -
            wait_penalty
        )

        # CRITICAL: Ensure score is NEVER 0.0 or 1.0
        if score <= 0.0:
            score = 0.001
        elif score >= 1.0:
            score = 0.999

        # Cap at task max
        task_config = self.TASKS[self._current_task]
        max_score = task_config["optimal_score"]

        final_score = min(score, max_score)

        # Final safety check
        if final_score <= 0.0:
            final_score = 0.001
        if final_score >= 1.0:
            final_score = 0.999

        return final_score
    
    def get_task_score(self) -> float:
        """Public method to get task score for grading."""
        score = self._calculate_score()
        # Ensure strictly between 0 and 1
        if score <= 0.0:
            return 0.001
        if score >= 1.0:
            return 0.999
        return score