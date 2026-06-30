#!/usr/bin/env python3
"""
Heterogeneous Multi-Agent Swarm -  Working Demo

This demo implements a dynamic local multi-agent swarm using different open-source models:
- Orchestrator (phi3): Dynamically routes tasks between agents based on conversation state.
- Cooperative agent (qwen3.5:4b): Collaborative problem-solver and coder.
- Adversarial agent (deepseek-coder:6.7b): Critical thinker who challenges assumptions.

The agents interact in a dynamic loop, storing conversation history in a FAISS vector 
database and evaluating outputs using real code quality checks (pylint).

Usage:
    python demo.py

Requirements:
    pip install ollama sentence-transformers faiss-cpu redis numpy pylint
"""

import os
import time
import json
import hashlib
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple

# For local LLM access
import ollama

# For vector storage and retrieval
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# For caching
import redis

class Colors:
    ORCHESTRATOR = '\033[95m' # Magenta
    COOPERATIVE = '\033[94m'  # Blue
    ADVERSARIAL = '\033[91m'  # Red
    SYSTEM = '\033[93m'       # Yellow
    RESET = '\033[0m'

# --- Configuration ---
REDIS_HOST = "localhost"
REDIS_PORT = 6379
ORCHESTRATOR_MODEL = "phi3:mini"
COOPERATIVE_MODEL = "qwen3.5:4b"
ADVERSARIAL_MODEL = "deepseek-coder:6.7b"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_INDEX_PATH = "./agent_vector_index.faiss"

# Personality templates for agents
PERSONALITY_PROMPTS = {
    "adversarial": (
        "You are an Adversarial AI agent, a critical thinker who challenges assumptions. "
        "Your job is to identify flaws, edge cases, and weaknesses in logic in the provided code. "
        "Provide constructive criticism and suggest improvements, but do not write the full solution yourself. "
        "If the code looks perfect and handles all edge cases, explicitly state 'The code looks good, no further flaws'."
    ),
    "cooperative": (
        "You are a Cooperative AI agent, a collaborative problem-solver. "
        "Your job is to write clean, efficient, and well-documented Python code. "
        "When given feedback, you adapt and refine your code to address all edge cases and flaws. "
        "Always enclose your final Python code in ```python ... ``` markdown blocks."
    )
}

# --- Core Logic ---

def extract_code(text: str) -> str:
    """Extracts python code from markdown block if present."""
    match = re.search(r'```(?:python)?\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        return match.group(1)
    return text


class VectorHistoryManager:
    """Manages conversation history with FAISS vector storage"""
    
    def __init__(self, index_path: str, embedding_model_name: str):
        self.index_path = index_path
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.documents = []  # Stores tuple of (agent_name, role, message, timestamp)
        self._load_or_create_index()

    def _load_or_create_index(self):
        try:
            self.index = faiss.read_index(self.index_path)
        except Exception:
            self.index = faiss.IndexFlatIP(384)

    def add_message(self, agent_name: str, message: str, role: str = "user") -> None:
        if not message.strip():
            return
            
        timestamp = datetime.now().isoformat()
        doc_tuple = (agent_name, role, message, timestamp)
        
        embedding = self.embedding_model.encode([message])
        embedding = np.array(embedding, dtype=np.float32)
        faiss.normalize_L2(embedding)
        
        self.index.add(embedding)
        self.documents.append(doc_tuple)
    
    def search_messages(self, query: str, top_k: int = 3) -> List[Tuple]:
        if self.index.ntotal == 0:
            return []
            
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx])
        
        return results


class Orchestrator:
    """Dynamically routes tasks based on conversation state."""
    
    def __init__(self, model_name: str = ORCHESTRATOR_MODEL):
        self.model_name = model_name
        self.prompt = (
            "You are the Orchestrator of a multi-agent swarm. Your job is to decide which agent should act next based on the history.\n"
            "Agents:\n"
            "- COOPERATIVE: Writes and refines code.\n"
            "- ADVERSARIAL: Critiques code and finds bugs.\n\n"
            "Rules:\n"
            "1. If the last agent to speak was 'system' or the history is empty, you MUST route to COOPERATIVE.\n"
            "2. If the last agent to speak was 'Cooperative', you MUST route to ADVERSARIAL. Do NOT evaluate the code yourself.\n"
            "3. If the last agent to speak was 'Adversarial' and they found flaws, you MUST route to COOPERATIVE.\n"
            "4. If the last agent to speak was 'Adversarial' and they explicitly state 'The code looks good', route to FINISHED.\n"
            "CRITICAL: Never route to FINISHED unless the ADVERSARIAL agent has reviewed the code and approved it.\n\n"
            "Output ONLY a valid JSON object matching this schema: {\"next_agent\": \"COOPERATIVE\" | \"ADVERSARIAL\" | \"FINISHED\", \"reason\": \"<your reasoning>\"}"
        )
        
    def decide(self, task_prompt: str, history: List[Tuple]) -> Dict:
        history_str = ""
        # Only feed the last 3 turns to keep context tight for the small model
        recent_history = history[-3:] if len(history) > 3 else history
        
        for doc in recent_history:
            msg = doc[2]
            if len(msg) > 600:
                msg = msg[:600] + "... [TRUNCATED]"
            history_str += f"Agent '{doc[0]}': {msg}\n\n"
            
        full_prompt = f"{self.prompt}\n\nOriginal Task: {task_prompt}\n\nRecent History:\n{history_str}\n\nJSON output:"
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=full_prompt,
                stream=False,
                format="json"
            )
            resp_text = response.get("response", "{}")
            
            # Extract JSON if wrapped in markdown
            match = re.search(r'\{.*\}', resp_text, re.DOTALL)
            if match:
                resp_text = match.group(0)
                
            return json.loads(resp_text)
        except Exception as e:
            print(f"Orchestrator error: {e}")
            return {"next_agent": "FINISHED", "reason": "Error parsing JSON."}


@dataclass
class Agent:
    """Represents a personality-based LLM agent"""
    name: str
    personality: str
    prompt_template: str
    model_name: str
    
    def get_response(self, task_prompt: str, context: str = "") -> str:
        full_prompt = f"{self.prompt_template}\n\n"
        if context:
            full_prompt += f"Previous Conversation Context:\n{context}\n\n"
        full_prompt += f"Current Task:\n{task_prompt}"
        
        try:
            stream = ollama.generate(
                model=self.model_name,
                prompt=full_prompt,
                stream=True
            )
            full_response = ""
            for chunk in stream:
                token = chunk.get("response", "")
                print(token, end="", flush=True)
                full_response += token
            return full_response
        except Exception as e:
            print(f"Exception in LLM call for {self.name}: {e}")
            return ""


class CodeEvaluator:
    """Evaluates code quality"""
    
    def evaluate_code_quality(self, code_snippet: str) -> Dict[str, float]:
        code = extract_code(code_snippet)
        temp_file = "temp_eval.py"
        with open(temp_file, "w") as f:
            f.write(code)
        
        try:
            result = subprocess.run(["pylint", temp_file], capture_output=True, text=True)
            match = re.search(r"rated at ([\d\.\-]+)/10", result.stdout)
            score = float(match.group(1)) if match else 0.0
        except Exception as e:
            print(f"Error evaluating code: {e}")
            score = 0.0
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        return {"quality_score": score}


class HeterogeneousSwarm:
    """Manages a dynamic swarm of heterogeneous agents"""
    def __init__(self, agents: List[Agent], orchestrator: Orchestrator, history: VectorHistoryManager, evaluator: CodeEvaluator, cache: redis.Redis):
        self.agents = {a.name.upper(): a for a in agents}
        self.orchestrator = orchestrator
        self.conversation_history = history
        self.evaluator = evaluator
        self.cache = cache
    
    def run_dynamic_workflow(self, task_prompt: str, max_turns: int = 5) -> Dict[str, List[str]]:
        print(f"Starting true multi-agent swarm interaction for task: {task_prompt}")
        
        self.conversation_history.add_message("system", task_prompt, "task")
        results = {name: [] for name in self.agents.keys()}
        
        turn = 0
        while turn < max_turns:
            print(f"\n--- Turn {turn + 1}: Orchestrator deciding... ---")
            
            # Orchestrator decides
            decision = self.orchestrator.decide(task_prompt, self.conversation_history.documents)
            next_agent_name = decision.get("next_agent", "FINISHED").upper()
            print(f"{Colors.ORCHESTRATOR}Orchestrator [{self.orchestrator.model_name}] Decision: Route to {next_agent_name}")
            print(f"Reasoning: {decision.get('reason', 'N/A')}{Colors.RESET}")
            
            if next_agent_name == "FINISHED" or next_agent_name not in self.agents:
                print("Swarm execution finished by Orchestrator.")
                break
                
            agent = self.agents[next_agent_name]
            
            # Build context via FAISS vector search
            if next_agent_name == "COOPERATIVE":
                search_query = "critique edge cases feedback flaws bugs"
                action_prompt = "Draft or refine the Python code based on the task and any recent feedback."
            else:
                search_query = "code implementation python function sequence"
                action_prompt = "Review the latest drafted code. Point out edge cases or bugs. If it is perfect, state 'The code looks good, no further flaws'."
                
            context_docs = self.conversation_history.search_messages(search_query, top_k=2)
            context_str = "\n".join([f"{doc[0]}: {doc[2]}" for doc in context_docs])
            
            full_prompt = f"{action_prompt}\nTask: {task_prompt}"
            cache_key = f"T{turn}_{agent.name}_{hashlib.md5(full_prompt.encode()).hexdigest()}"
            
            # Safe cache get
            resp = None
            try:
                resp = self.cache.get(cache_key)
            except redis.exceptions.ConnectionError:
                pass # Redis is down, skip cache
                
            color = Colors.COOPERATIVE if agent.name.upper() == "COOPERATIVE" else Colors.ADVERSARIAL
                
            if resp:
                print(f"Using cached response from {agent.name}")
                print(f"\n{color}{agent.name} [{agent.model_name}]:\n{resp}{Colors.RESET}\n")
            else:
                print(f"\n{color}{agent.name} [{agent.model_name}] is generating response:\n", end="", flush=True)
                resp = agent.get_response(full_prompt, context=context_str)
                print(f"{Colors.RESET}\n")
                
                # Safe cache set
                try:
                    self.cache.setex(cache_key, 3600, resp)
                except redis.exceptions.ConnectionError:
                    pass # Redis is down, skip cache
                
            self.conversation_history.add_message(agent.name, resp)
            results[agent.name.upper()].append(resp)
            
            turn += 1
            
        return results

    def evaluate_final_output(self, results: Dict[str, List[str]]):
        print("\n--- Evaluating Final Output ---")
        coop_results = results.get("COOPERATIVE", [])
        if not coop_results:
            print("No code was generated by the Cooperative agent.")
            return None
            
        final_code = coop_results[-1]
        quality = self.evaluator.evaluate_code_quality(final_code)
        print(f"Final Code Pylint Score: {quality['quality_score']:.2f} / 10.0")
        
        return quality


def main() -> None:
    print("Starting HETEROGENEOUS Multi-Agent Swarm Demo")
    print("=====================================================")
    
    task_prompt = (
        "Write a Python class that implements an LRU (Least Recently Used) Cache. "
        "It should support get() and put() operations in O(1) time complexity. "
        "Ensure edge cases like 0 capacity, negative capacity, or updating existing keys are handled."
    )
    print(f"\n{Colors.SYSTEM}Starting true multi-agent swarm interaction for task:\n{task_prompt}{Colors.RESET}\n")
    
    try:
        # Setup Redis cache
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        
        # Create agents with specific models
        adversarial_agent = Agent(
            name="Adversarial",
            personality="adversarial",
            prompt_template=PERSONALITY_PROMPTS["adversarial"],
            model_name=ADVERSARIAL_MODEL
        )
        cooperative_agent = Agent(
            name="Cooperative",
            personality="cooperative",
            prompt_template=PERSONALITY_PROMPTS["cooperative"],
            model_name=COOPERATIVE_MODEL
        )
        
        orchestrator = Orchestrator(model_name=ORCHESTRATOR_MODEL)
        history_manager = VectorHistoryManager(VECTOR_INDEX_PATH, EMBEDDING_MODEL)
        evaluator = CodeEvaluator()
        
        swarm = HeterogeneousSwarm(
            agents=[adversarial_agent, cooperative_agent],
            orchestrator=orchestrator,
            history=history_manager,
            evaluator=evaluator,
            cache=redis_client
        )
        

        
        results = swarm.run_dynamic_workflow(task_prompt, max_turns=6)
        swarm.evaluate_final_output(results)
        
    except Exception as e:
        print(f"Error in demo execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
