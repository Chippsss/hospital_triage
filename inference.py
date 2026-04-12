"""
Inference Script for Hospital Triage Environment
Uses OpenAI API with fallback to wait() on errors.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
import textwrap
import json
import re
from typing import List, Optional

from openai import OpenAI

from client import HospitalTriageEnv
from models import HospitalTriageAction, ActionType

# Environment variables (set by the validator or judge)
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
BENCHMARK = "hospital_triage"
MAX_STEPS = 30
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.7

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a hospital triage manager. Your goal is to treat all patients efficiently.
    
    Available actions:
    1. assign_doctor(patient_id, doctor_id) - Assign a specific doctor to a patient
    2. wait() - Do nothing for this step
    
    Strategy:
    - Prioritize critical patients (they appear in critical_patients list)
    - Use all available doctors (free_doctors list)
    
    Respond with a JSON action in this exact format:
    {"action_type": "assign_doctor", "patient_id": 0, "doctor_id": 0}
    OR
    {"action_type": "wait"}
    
    Only respond with the JSON, no other text.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_llm_action(client: Optional[OpenAI], observation, step: int, last_reward: float, history: List[str]) -> HospitalTriageAction:
    """
    Get action from OpenAI LLM. If client is None or API call fails, return a safe fallback (wait).
    """
    # If no client (no API key), return wait
    if client is None:
        return HospitalTriageAction(action_type=ActionType.WAIT)

    user_prompt = textwrap.dedent(
        f"""
        Step: {step}/{MAX_STEPS}
        Waiting patients: {observation.waiting_patients}
        Critical patients: {observation.critical_patients}
        Free doctors: {observation.free_doctors}
        Patients treated: {observation.treated_ratio * 100:.0f}%
        Last reward: {last_reward:.2f}
        Choose your next action (JSON only):
        """
    ).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = completion.choices[0].message.content or ""
        json_match = re.search(r'\{.*?\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if data.get("action_type") == "assign_doctor":
                return HospitalTriageAction(
                    action_type=ActionType.ASSIGN_DOCTOR,
                    patient_id=data.get("patient_id"),
                    doctor_id=data.get("doctor_id")
                )
            elif data.get("action_type") == "wait":
                return HospitalTriageAction(action_type=ActionType.WAIT)
    except Exception as e:
        # Log error but do not crash – return a safe wait action
        print(f"[WARN] LLM call failed: {e}. Using wait action.", flush=True)

    # Fallback: wait
    return HospitalTriageAction(action_type=ActionType.WAIT)


async def run_task(env: HospitalTriageEnv, client: Optional[OpenAI], task_id: str, task_name: str) -> tuple:
    rewards = []
    steps_taken = 0
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME if client else "no-llm-fallback")
    
    try:
        result = await env.reset(task_id=task_id)
        observation = result.observation
        last_reward = 0.0
        history = []
        
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break
            action = get_llm_action(client, observation, step, last_reward, history)
            result = await env.step(action)
            observation = result.observation
            reward = result.reward or 0.0
            done = result.done
            rewards.append(reward)
            steps_taken = step
            last_reward = reward
            action_str = f"assign_doctor({action.patient_id},{action.doctor_id})" if action.patient_id else "wait()"
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)
            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")
            if done:
                break
        
        state = await env.state()
        if state.total_patients > 0:
            treatment_score = state.treated_patients / state.total_patients
            critical_score = state.critical_treated / state.total_critical if state.total_critical > 0 else 1.0
            efficiency = min(state.efficiency, 1.0)
            score = treatment_score * 0.5 + critical_score * 0.3 + efficiency * 0.2
        else:
            score = 0.0
        score = min(max(score, 0.001), 0.999)
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as e:
        print(f"Task error: {e}", flush=True)
        score, success = 0.0, False
    
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


async def main():
    print("="*60)
    print("Hospital Triage - Inference")
    print("="*60)
    
    # Initialize OpenAI client only if API key is present
    client = None
    if API_KEY:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        print(f"Using LLM: {MODEL_NAME}")
    else:
        print("No OpenAI API key provided. Using fallback actions (wait only).")
    
    base_url = os.getenv("ENV_URL", "http://localhost:8000")
    async with HospitalTriageEnv(base_url=base_url) as env:
        tasks = [
            ("easy_triage", "Basic Triage"),
            ("medium_triage", "Medium Hospital Management"),
            ("hard_triage", "Emergency Overload")
        ]
        all_scores = {}
        for task_id, task_name in tasks:
            print(f"\n--- {task_name} ---")
            scores = []
            for run in range(3):
                print(f"\nRun {run+1}:")
                score = await run_task(env, client, task_id, task_name)
                scores.append(score)
                print(f"Score: {score:.3f}")
            avg = sum(scores)/len(scores)
            all_scores[task_id] = avg
            print(f"Average: {avg:.3f}")
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    for k,v in all_scores.items():
        print(f"{k}: {v:.3f}")
    with open("baseline_scores.json","w") as f:
        json.dump(all_scores, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())