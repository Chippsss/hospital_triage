"""
Inference Script for Hospital Triage Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    OPENAI_API_KEY Your OpenAI API key.

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
"""

import asyncio
import os
import textwrap
import json
import re
from typing import List, Optional

from openai import OpenAI

from hospital_triage.client import HospitalTriageEnv
from hospital_triage.models import HospitalTriageAction, ActionType

# Environment variables (OpenAI only)
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
TASK_NAME = os.getenv("HOSPITAL_TRIAGE_TASK", "easy_triage")
BENCHMARK = os.getenv("HOSPITAL_TRIAGE_BENCHMARK", "hospital_triage")
MAX_STEPS = 30
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.7

MAX_TOTAL_REWARD = 100.0

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a hospital triage manager. Your goal is to treat all patients efficiently.
    
    Available actions:
    1. assign_doctor(patient_id, doctor_id) - Assign a specific doctor to a patient
    2. wait() - Do nothing for this step
    
    Strategy:
    - Prioritize critical patients (they appear in critical_patients list)
    - Use all available doctors (free_doctors list)
    - Each treated patient gives reward +1, critical patients give +3 total
    
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
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, observation, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}/{MAX_STEPS}
        Waiting patients: {observation.waiting_patients}
        Critical patients: {observation.critical_patients}
        Free doctors: {observation.free_doctors}
        Patients treated: {observation.treated_ratio * 100:.0f}%
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Choose your next action (JSON only):
        """
    ).strip()


def get_model_message(client: OpenAI, step: int, observation, last_reward: float, history: List[str]) -> HospitalTriageAction:
    """Get action from OpenAI LLM and parse into HospitalTriageAction"""
    
    user_prompt = build_user_prompt(step, observation, last_reward, history)
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        # Parse JSON response
        json_match = re.search(r'\{.*?\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            action_type = data.get("action_type", "wait")
            
            if action_type == "assign_doctor":
                return HospitalTriageAction(
                    action_type=ActionType.ASSIGN_DOCTOR,
                    patient_id=data.get("patient_id"),
                    doctor_id=data.get("doctor_id")
                )
        
        return HospitalTriageAction(action_type=ActionType.WAIT)
        
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return HospitalTriageAction(action_type=ActionType.WAIT)


async def run_task(env: HospitalTriageEnv, client: OpenAI, task_id: str) -> tuple:
    """Run a single task episode"""
    
    task_names = {
        "easy_triage": "Basic Triage",
        "medium_triage": "Medium Hospital Management",
        "hard_triage": "Emergency Overload"
    }
    
    task_name = task_names.get(task_id, task_id)
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        result = await env.reset(task_id=task_id)
        observation = result.observation
        last_reward = 0.0
        
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break
            
            action = get_model_message(client, step, observation, last_reward, history)
            
            result = await env.step(action)
            observation = result.observation
            
            reward = result.reward or 0.0
            done = result.done
            error = None
            
            rewards.append(reward)
            steps_taken = step
            last_reward = reward
            
            # Format action string for logging
            if action.action_type == ActionType.ASSIGN_DOCTOR:
                action_str = f"assign_doctor({action.patient_id},{action.doctor_id})"
            else:
                action_str = "wait()"
            
            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error
            )
            
            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")
            
            if done:
                break
        
        # Calculate final score (normalized 0-1)
        state = await env.state()
        if state.total_patients > 0:
            treatment_score = state.treated_patients / state.total_patients
            critical_score = (
                state.critical_treated / state.total_critical 
                if state.total_critical > 0 else 1.0
            )
            efficiency = min(state.efficiency, 1.0)
            score = treatment_score * 0.5 + critical_score * 0.3 + efficiency * 0.2
        else:
            score = 0.0
        
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
        
    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)
        score = 0.0
        success = False
    
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score, rewards, steps_taken, success


async def main() -> None:
    """Main inference loop"""
    
    # Check for API key
    if not API_KEY:
        print("[DEBUG] ERROR: OPENAI_API_KEY environment variable not set!", flush=True)
        print("[DEBUG] Please set your OpenAI API key and try again.", flush=True)
        return
    
    print("[DEBUG] Starting Hospital Triage Inference", flush=True)
    print(f"[DEBUG] API Base: {API_BASE_URL}", flush=True)
    print(f"[DEBUG] Model: {MODEL_NAME}", flush=True)
    
    # Initialize OpenAI client
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Connect to environment (local server)
    base_url = os.getenv("ENV_URL", "http://localhost:8000")
    
    async with HospitalTriageEnv(base_url=base_url) as env:
        tasks = ["easy_triage", "medium_triage", "hard_triage"]
        all_scores = {}
        
        for task_id in tasks:
            print(f"\n[DEBUG] Running {task_id}...", flush=True)
            score, rewards, steps, success = await run_task(env, client, task_id)
            all_scores[task_id] = score
            print(f"[DEBUG] {task_id} complete: score={score:.3f}, steps={steps}, success={success}\n", flush=True)
        
        # Final summary
        print("\n" + "=" * 50, flush=True)
        print("FINAL RESULTS", flush=True)
        print("=" * 50, flush=True)
        for task, score in all_scores.items():
            print(f"  {task}: {score:.3f}", flush=True)
        avg_score = sum(all_scores.values()) / len(all_scores)
        print(f"  Average: {avg_score:.3f}", flush=True)
        
        # Save results
        with open("baseline_scores.json", "w") as f:
            json.dump(all_scores, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())