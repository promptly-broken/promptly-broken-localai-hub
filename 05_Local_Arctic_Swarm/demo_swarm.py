#!/usr/bin/env python3
"""
ArcticSwarm True Asynchronous Demo (LLM-Driven Version)

A multi-agent system where each agent runs in its own thread and
listens to a Redis message queue. In this version, all agents use local
LLMs to make decisions and perform tasks.
"""

import time
import sys
import json
import threading
import uuid
import re
from typing import Dict, Any
import redis
import ollama
import faiss
from sentence_transformers import SentenceTransformer

# --- Configuration ---
REDIS_HOST = "localhost"
REDIS_PORT = 6379

ROUTING_MODEL = "qwen3.5:4b"    # Fast model for governance/coordination
RESEARCH_MODEL = "gemma4:26b"   # Heavyweight model for deep research

# Redis Queues
Q_INCOMING = "queue:incoming"
Q_GBB_CONSENSUS = "queue:gbb_consensus"
Q_GBB_DELEGATED = "queue:gbb_delegated"
Q_GBB_MERITOCRATIC = "queue:gbb_meritocratic"
Q_VECTOR_SEARCH = "queue:vector_search"
Q_LLM_RESEARCH = "queue:llm_research"
Q_RESULTS = "queue:results"

def get_redis() -> redis.Redis:
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'

# --- Realistic Document Corpus ---
DOCUMENTS = [
    """The Evolution of Machine Learning: 
Historically, machine learning was defined as the field of study that gives computers the ability to learn without being explicitly programmed. Early ML relied heavily on feature engineering, where domain experts manually crafted inputs that would allow algorithms like Support Vector Machines (SVM) or Random Forests to classify data effectively. The transition to Deep Learning changed this paradigm. Deep learning models, composed of multiple layers of artificial neural networks, inherently learn representations of data with multiple levels of abstraction. This meant that for image or speech recognition, the network itself discovers the features, eliminating much of the manual engineering. However, Deep Learning models typically require vast amounts of labeled training data and computational resources to converge effectively.""",

    """Vector Databases and Retrieval-Augmented Generation (RAG):
A vector database is a specialized storage system designed to index and query high-dimensional vectors efficiently. When combined with large language models, they form the backbone of Retrieval-Augmented Generation (RAG) architectures. In a RAG setup, an incoming user query is first passed through an embedding model (like all-MiniLM-L6-v2) to generate a dense vector representation. This vector is then used to perform a nearest-neighbor search (using algorithms like HNSW or libraries like FAISS) against a pre-indexed corpus of documents. The retrieved documents are injected into the prompt of a generative LLM, granting it access to external, domain-specific knowledge that wasn't present in its original training weights, effectively reducing hallucinations and increasing factual accuracy.""",

    """Agentic Design Patterns in AI Systems:
Modern AI architectures are shifting from simple prompt-response loops to agentic systems. In an agentic pattern, an LLM is given a role, a set of tools, and an environment to interact with. Routing is a common pattern where an initial LLM acts as a triage agent, analyzing the intent of a message and routing it to a specialized sub-agent. For example, a multi-agent system might have a 'Governance Agent' that stamps permissions and routes queries into 'consensus' or 'delegated' queues. This strictly decoupled design allows each agent to operate asynchronously, often communicating over a message bus like Redis or Kafka. This mimics organizational structures, allowing for scalable, resilient AI swarms that can handle complex, multi-step workflows.""",

    """Local AI Ecosystems and Hardware Constraints:
Running AI models locally has become increasingly feasible thanks to tools like Ollama, LM Studio, and llama.cpp. These tools utilize techniques like quantization (reducing model weights to 4-bit or 8-bit precision) to drastically lower the memory requirements. For example, a massive 30-billion parameter model, which normally requires over 60GB of VRAM in float16, can be run on a MacBook with 32GB of unified memory when quantized to 4-bit. This has spurred a revolution in local-first AI, allowing developers to build enterprise-grade systems entirely on consumer hardware, ensuring data privacy and zero API costs."""
]

# --- Agents ---

class BaseAgent(threading.Thread):
    def __init__(self, name: str, listen_queue: str, color: str = Colors.RESET):
        super().__init__(daemon=True)
        self.name = name
        self.listen_queue = listen_queue
        self.color = color
        self.redis = get_redis()
        self.client = ollama.Client()
        self.running = True
        self.log(f"Initialized, listening on {self.listen_queue}")

    def log(self, text: str, newline=True):
        if newline:
            print(f"{self.color}[{self.name}] {text}{Colors.RESET}")
        else:
            print(f"{self.color}[{self.name}] {text}{Colors.RESET}", end="", flush=True)

    def run(self):
        self.log("Started.")
        while self.running:
            msg_data = self.redis.blpop(self.listen_queue, timeout=1)
            if msg_data:
                _, msg_json = msg_data
                try:
                    message = json.loads(msg_json)
                    self.log(f"\nPicked up message ID: {message.get('id', 'unknown')}")
                    self.process_message(message)
                except Exception as e:
                    self.log(f"Error processing message: {e}")
                    
    def process_message(self, message: Dict[str, Any]):
        raise NotImplementedError
        
    def stop(self):
        self.running = False


class GovernanceAgent(BaseAgent):
    """Uses LLM to analyze query intent and determine the governance mode."""
    def __init__(self):
        super().__init__("GovernanceAgent", Q_INCOMING, Colors.BLUE)
        
    def process_message(self, message: Dict[str, Any]):
        query = message.get("query", "")
        
        prompt = (
            f"You are a Governance AI. Analyze the following user request:\n"
            f"'{query}'\n\n"
            f"If it asks for facts, knowledge, or research, output 'consensus'.\n"
            f"If it asks for an opinion or creative work, output 'meritocratic'.\n"
            f"If it asks to execute an action, output 'delegated'.\n"
            f"Output ONLY the single word."
        )
        
        self.log(f"Asking {ROUTING_MODEL} for governance routing... ", newline=False)
        try:
            stream = self.client.generate(model=ROUTING_MODEL, prompt=prompt, stream=True)
            chunks = []
            for chunk in stream:
                part = chunk['response']
                chunks.append(part)
                print(f"{self.color}{part}{Colors.RESET}", end="", flush=True)
            print()
            mode = "".join(chunks).strip().lower()
            
            # Clean up the response in case the LLM was chatty
            if "consensus" in mode: mode = "consensus"
            elif "meritocratic" in mode: mode = "meritocratic"
            elif "delegated" in mode: mode = "delegated"
            else: mode = "consensus" # fallback
            
        except Exception as e:
            print()
            self.log(f"LLM Error: {e}. Falling back to consensus.")
            mode = "consensus"
            
        message["governed_by"] = mode
        message["timestamp"] = time.time()
        
        queue_map = {
            "consensus": Q_GBB_CONSENSUS,
            "delegated": Q_GBB_DELEGATED,
            "meritocratic": Q_GBB_MERITOCRATIC
        }
        target_queue = queue_map.get(mode, Q_GBB_CONSENSUS)
        
        self.log(f"Routed to '{mode}' governance. Forwarding to {target_queue}")
        self.redis.rpush(target_queue, json.dumps(message))


class CoordinatorAgent(BaseAgent):
    """Uses LLM to extract keywords for vector search optimization."""
    def __init__(self):
        super().__init__("CoordinatorAgent", Q_GBB_CONSENSUS, Colors.YELLOW)
        
    def process_message(self, message: Dict[str, Any]):
        query = message.get("query", "")
        
        prompt = (
            f"You are a search query optimizer.\n"
            f"Given the user's request: '{query}'\n"
            f"Extract the 3 most crucial keywords or short phrases for a vector database search.\n"
            f"Output ONLY a comma-separated list of keywords, nothing else."
        )
        
        self.log(f"Asking {ROUTING_MODEL} to extract search keywords... ", newline=False)
        try:
            stream = self.client.generate(model=ROUTING_MODEL, prompt=prompt, stream=True)
            chunks = []
            for chunk in stream:
                part = chunk['response']
                chunks.append(part)
                print(f"{self.color}{part}{Colors.RESET}", end="", flush=True)
            print()
            keywords = "".join(chunks).strip()
            # Clean up formatting
            keywords = re.sub(r'^[^\w]+', '', keywords) # remove leading punctuation
            message["search_keywords"] = keywords
        except Exception as e:
            print()
            self.log(f"LLM Error: {e}. Falling back to raw query.")
            message["search_keywords"] = query
            
        self.redis.rpush(Q_VECTOR_SEARCH, json.dumps(message))


class VectorStorageAgent(BaseAgent):
    """Embeds the keywords, queries FAISS, and forwards to Research Agent."""
    def __init__(self):
        super().__init__("VectorStorageAgent", Q_VECTOR_SEARCH, Colors.MAGENTA)
        self.log(f"Loading sentence-transformers...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatIP(384)
        
        self.log(f"Embedding {len(DOCUMENTS)} realistic documents...")
        embeddings = self.model.encode(DOCUMENTS)
        self.index.add(embeddings)
        self.log(f"FAISS index ready.")
        
    def process_message(self, message: Dict[str, Any]):
        search_query = message.get("search_keywords", message.get("query", ""))
        self.log(f"Searching FAISS for: '{search_query}'")
        
        query_emb = self.model.encode([search_query])
        distances, indices = self.index.search(query_emb, k=2)
        
        context = [DOCUMENTS[i] for i in indices[0]]
        message["context"] = context
        self.log(f"Retrieved 2 dense context chunks. Forwarding to LLM.")
        
        self.redis.rpush(Q_LLM_RESEARCH, json.dumps(message))


class ResearchAgent(BaseAgent):
    """Uses heavy LLM to perform deep research based on provided FAISS context."""
    def __init__(self):
        super().__init__("ResearchAgent", Q_LLM_RESEARCH, Colors.GREEN)
        
    def process_message(self, message: Dict[str, Any]):
        query = message.get("query", "")
        context = message.get("context", [])
        
        context_str = "\n\n---\n\n".join(context)
        prompt = (
            f"You are a Senior AI Research Analyst. Answer the following question thoroughly based on the provided context.\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"QUESTION: {query}\n\n"
            f"DETAILED ANSWER:"
        )
        
        self.log(f"Generating deep research response using {RESEARCH_MODEL} (streaming)...")
        self.log("Response: ", newline=False)
        try:
            stream = self.client.generate(model=RESEARCH_MODEL, prompt=prompt, stream=True)
            answer_chunks = []
            for chunk in stream:
                part = chunk['response']
                answer_chunks.append(part)
                print(f"{self.color}{part}{Colors.RESET}", end="", flush=True)
            print()
            message["answer"] = "".join(answer_chunks)
        except Exception as e:
            print()
            self.log(f"Ollama Error: {e}")
            message["answer"] = f"Error generating answer: {e}"
            
        self.log(f"Answer generated. Forwarding to results.")
        self.redis.rpush(Q_RESULTS, json.dumps(message))


# --- Main ---

def flush_queues(r: redis.Redis):
    for q in [Q_INCOMING, Q_GBB_CONSENSUS, Q_GBB_DELEGATED, Q_GBB_MERITOCRATIC, Q_VECTOR_SEARCH, Q_LLM_RESEARCH, Q_RESULTS]:
        r.delete(q)

def main():
    print("=== Starting LLM-Driven ArcticSwarm Async Demo ===")
    r = get_redis()
    
    try:
        r.ping()
        print("Connected to Redis successfully.")
    except redis.ConnectionError:
        print("CRITICAL: Cannot connect to Redis on localhost:6379.")
        return

    flush_queues(r)

    # Initialize and start agents
    agents = [
        GovernanceAgent(),
        CoordinatorAgent(),
        VectorStorageAgent(),
        ResearchAgent()
    ]
    
    for agent in agents:
        agent.start()
        
    time.sleep(2)
    print("\n--- All Agents Running in Background Threads ---")
    
    queries = [
        "Can you explain how Retrieval-Augmented Generation relies on vector databases and embedding models?"
    ]
    
    for query in queries:
        msg_id = str(uuid.uuid4())[:8]
        msg = {
            "id": msg_id,
            "query": query,
            "sender": "user"
        }
        print(f"\n{Colors.CYAN}[User] Submitting query '{msg_id}': {query}{Colors.RESET}")
        r.rpush(Q_INCOMING, json.dumps(msg))
        
        print(f"{Colors.CYAN}[User] Waiting for result on queue '{Q_RESULTS}'...{Colors.RESET}")
        result_data = r.blpop(Q_RESULTS, timeout=120)
        
        if result_data:
            _, result_json = result_data
            result = json.loads(result_json)
            print("\n" + "="*80)
            print(f"{Colors.CYAN}FINAL RESULT for '{result['id']}'{Colors.RESET}")
            print(f"Governed By: {result.get('governed_by')}")
            print(f"Extracted Keywords: {result.get('search_keywords')}")
            print(f"Answer:\n{result.get('answer')}")
            print("="*80 + "\n")
        else:
            print(f"\n{Colors.CYAN}[User] TIMEOUT waiting for result for query '{msg_id}'!{Colors.RESET}")
            
    print("Demo complete. Shutting down agents.")
    for agent in agents:
        agent.stop()
    for agent in agents:
        agent.join(timeout=1.0)
    print("All agents stopped.")

if __name__ == "__main__":
    main()
