#!/usr/bin/env python3
"""
Stateless MCP Server Demo — Promptly Broken

A production-grade Model Context Protocol (MCP) server that exposes real
local tools (FAISS Vector Search, System Profiler, Web Scraper). Includes a 
Redis-backed stateful baseline for head-to-head latency and memory benchmarking.

Usage:
    python demo.py

Requirements:
    pip install fastapi uvicorn requests pydantic psutil faiss-cpu sentence-transformers redis beautifulsoup4

Hardware: Apple M-series with 48GB unified memory (MPS/CPU)
"""

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional
import glob

import psutil
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import faiss
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

os.environ["HF_HUB_OFFLINE"] = "1"

# --- Configuration ---
STATELESS_PORT = 8100
STATEFUL_PORT = 8101
TOOLS_CONFIG_PATH = "tools_config.json"
VAULT_PATH = "obsidian_vault"
REDIS_HOST = "localhost"
REDIS_PORT = 6379

# --- Colors ---
C_SL = '\033[96m'  # Cyan for Stateless
C_SF = '\033[92m'  # Green for Stateful
C_RES = '\033[0m'  # Reset

# --- Data Models ---
class ToolCallRequest(BaseModel):
    id: str
    method: str
    params: Dict[str, Any]


class ToolCallResponse(BaseModel):
    id: str
    result: Dict[str, Any]


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


# ═══════════════════════════════════════════════════════════
# Local Tools
# ═══════════════════════════════════════════════════════════

class LocalToolExecutor:
    """Executes actual Python code for the MCP tools locally."""
    
    def __init__(self):
        print("  Initializing FAISS Vault Searcher (loading SentenceTransformer)...")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.documents = []
        self._build_index()

    def _build_index(self):
        if not os.path.exists(VAULT_PATH):
            os.makedirs(VAULT_PATH)
            # Create dummy files if missing
            with open(os.path.join(VAULT_PATH, "dummy.md"), "w") as f:
                f.write("# Dummy Note\nThis is just a placeholder.")
                
        md_files = glob.glob(os.path.join(VAULT_PATH, "*.md"))
        if not md_files:
            print("  No markdown files found in vault.")
            return

        texts = []
        for file in md_files:
            with open(file, "r") as f:
                content = f.read().strip()
                if content:
                    texts.append(f"[{os.path.basename(file)}] {content}")
        
        if texts:
            embeddings = self.encoder.encode(texts)
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(embeddings)
            self.documents = texts
            print(f"  ✓ Indexed {len(texts)} documents into FAISS.")

    def search_vault(self, query: str) -> str:
        """Execute FAISS semantic search (Local ML CPU Bound)."""
        if not self.index or not self.documents:
            return "Vault is empty or unindexed."
            
        q_emb = self.encoder.encode([query])
        distances, indices = self.index.search(q_emb, k=1)
        idx = indices[0][0]
        if idx != -1:
            return self.documents[idx]
        return "No relevant notes found."

    def get_system_stats(self) -> Dict[str, Any]:
        """Execute system profiling (Fast Local CPU Bound)."""
        mem = psutil.virtual_memory()
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "ram_used_gb": round(mem.used / (1024**3), 2),
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "ram_percent": mem.percent
        }
        
    def fetch_webpage(self, url: str) -> str:
        """Fetch and extract text from a webpage (Network I/O Bound)."""
        try:
            headers = {"User-Agent": "PromptlyBroken/1.0"}
            resp = requests.get(url, headers=headers, timeout=5)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            # Extract clean text, trim to 500 chars for demo
            text = soup.get_text(separator=' ', strip=True)
            return text[:500] + ("..." if len(text) > 500 else "")
        except Exception as e:
            return f"Error fetching webpage: {str(e)}"
        
    def execute(self, tool_name: str, params: Dict[str, Any]) -> Any:
        if tool_name == "search_vault":
            return {"retrieved_context": self.search_vault(params.get("query", ""))}
        elif tool_name == "get_system_stats":
            return self.get_system_stats()
        elif tool_name == "fetch_webpage":
            return {"content": self.fetch_webpage(params.get("url", ""))}
        else:
            raise ValueError(f"Tool {tool_name} not implemented locally.")


# ═══════════════════════════════════════════════════════════
# MCP Server Components
# ═══════════════════════════════════════════════════════════

class StatelessToolRegistry:
    """Manages tool definitions loaded from a config file. Zero session state."""
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.tools: Dict[str, ToolDefinition] = {}
        self._load_tools()

    def _load_tools(self):
        try:
            with open(self.config_path, "r") as f:
                config_data = json.load(f)
            for tool_def in config_data.get("tools", []):
                self.tools[tool_def["name"]] = ToolDefinition(**tool_def)
            print(f"  Loaded {len(self.tools)} tools from {self.config_path}")
        except FileNotFoundError:
            create_sample_config()
            self._load_tools()

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())


class StatefulSessionStore:
    """Per-session state using Redis — realistic architecture for scaling."""
    def __init__(self):
        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        # Test connection
        self.redis.ping()
        self.redis.flushdb()  # Clear for benchmark

    def record_call(self, session_id: str, call_data: Dict[str, Any]):
        """Append call context to a Redis list simulating session memory."""
        key = f"session:{session_id}:history"
        payload = json.dumps(call_data)
        self.redis.rpush(key, payload)
        # Expire session after 1 hour
        self.redis.expire(key, 3600)

    @property
    def total_bloat_kb(self) -> float:
        """Calculate total size of all session keys in Redis."""
        total_bytes = 0
        for key in self.redis.keys("session:*"):
            items = self.redis.lrange(key, 0, -1)
            total_bytes += sum(len(i) for i in items)
        return total_bytes / 1024


# ── Server Builders ──

def build_stateless_app(registry: StatelessToolRegistry, executor: LocalToolExecutor) -> FastAPI:
    app = FastAPI(title="Stateless MCP Server")

    @app.post("/mcp/v1/tool_call", response_model=ToolCallResponse)
    async def handle_tool_call(request: ToolCallRequest):
        if request.method != "tool_call":
            raise HTTPException(400, "Unsupported method")
            
        tool_name = request.params.get("tool_name")
        params = request.params.get("params", {})
        
        if not registry.get_tool(tool_name):
            raise HTTPException(404, f"Tool '{tool_name}' not found in registry")

        # Execute tool locally (no LLM in the loop here, we ARE the tool provider)
        result = executor.execute(tool_name, params)
        return ToolCallResponse(id=request.id, result=result)

    return app


def build_stateful_app(
    registry: StatelessToolRegistry, 
    executor: LocalToolExecutor,
    store: StatefulSessionStore
) -> FastAPI:
    app = FastAPI(title="Stateful MCP Server")

    @app.post("/mcp/v1/tool_call", response_model=ToolCallResponse)
    async def handle_tool_call(request: ToolCallRequest):
        tool_name = request.params.get("tool_name")
        params = request.params.get("params", {})
        session_id = request.params.get("session_id", request.id)
        
        if not registry.get_tool(tool_name):
            raise HTTPException(404, "Tool not found")

        # Execute tool locally
        result = executor.execute(tool_name, params)

        # STATEFUL OVERHEAD: Write to Redis
        store.record_call(session_id, {
            "tool": tool_name,
            "params": params,
            "result": result,
            "timestamp": time.time()
        })

        return ToolCallResponse(id=request.id, result=result)

    return app


# ═══════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════

def send_tool_call(
    port: int, tool_name: str, params: Dict[str, Any], call_id: str = "bench"
) -> tuple[Dict[str, Any], float]:
    payload = {
        "id": call_id,
        "method": "tool_call",
        "params": {"tool_name": tool_name, "params": params, "session_id": call_id},
    }
    start = time.perf_counter()
    resp = requests.post(f"http://127.0.0.1:{port}/mcp/v1/tool_call", json=payload, timeout=30)
    latency_ms = (time.perf_counter() - start) * 1000
    resp.raise_for_status()
    return resp.json(), latency_ms


def benchmark_latency(
    port: int, label: str, tool_name: str, params: Dict[str, Any], rounds: int = 5, color: str = ""
) -> List[float]:
    latencies = []
    for i in range(rounds):
        _, lat = send_tool_call(port, tool_name, params, call_id=f"{label}-seq-{i}")
        latencies.append(lat)
        print(f"    {color}{label} round {i + 1}/{rounds}: {lat:.1f} ms{C_RES}")
    return latencies


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def create_sample_config():
    config = {
        "tools": [
            {
                "name": "search_vault",
                "description": "Semantic search over local Obsidian notes using FAISS.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_system_stats",
                "description": "Get current MacBook CPU and RAM usage.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "fetch_webpage",
                "description": "Fetch text content from a remote URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to scrape"}
                    },
                    "required": ["url"],
                },
            },
        ]
    }
    with open(TOOLS_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Created tools config: {TOOLS_CONFIG_PATH}")


def summarize_with_llm(text: str) -> str:
    """Summarizes scraped text using a local Ollama model."""
    url = "http://localhost:11434/v1/chat/completions"
    payload = {
        "model": "gemma3:270m",
        "messages": [{"role": "user", "content": f"Summarize this in one sentence: {text[:2000]}"}],
        "stream": False
    }
    try:
        start = time.perf_counter()
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        summary = resp.json()["choices"][0]["message"]["content"].strip()
        lat = (time.perf_counter() - start) * 1000
        return f"{summary} (LLM latency: {lat:.1f} ms)"
    except Exception as e:
        return f"Failed to summarize (is Ollama running?): {e}"


def start_server(app: FastAPI, port: int, name: str):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name=name)
    thread.start()
    for _ in range(30):
        try:
            requests.get(f"http://127.0.0.1:{port}/docs", timeout=1)
            return server
        except Exception:
            time.sleep(0.2)
    raise RuntimeError(f"{name} failed to start on port {port}")


# ═══════════════════════════════════════════════════════════
# Main Demo
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Stateless MCP Server Demo — FAISS + Network + Redis")
    print("=" * 60)

    # ── Step 1: Setup ─────────────────────────────────────
    print("\n[1/5] Setup")
    # Always regenerate config for demo to include latest tools
    create_sample_config()

    registry = StatelessToolRegistry(TOOLS_CONFIG_PATH)
    executor = LocalToolExecutor()
    
    try:
        session_store = StatefulSessionStore()
        print("  ✓ Redis connected (baseline state store)")
    except redis.exceptions.ConnectionError:
        print("  ✗ Redis is not running. Please start the docker container on port 6379.")
        return

    # ── Step 2: Start servers ─────────────────────────────
    print("\n[2/5] Starting servers")
    stateless_app = build_stateless_app(registry, executor)
    stateful_app = build_stateful_app(registry, executor, session_store)

    start_server(stateless_app, STATELESS_PORT, "stateless")
    print(f"  {C_SL}✓ Stateless MCP server on port {STATELESS_PORT}{C_RES}")
    start_server(stateful_app, STATEFUL_PORT, "stateful")
    print(f"  {C_SF}✓ Stateful  MCP server on port {STATEFUL_PORT}{C_RES}")

    # ── Step 3: Tool calls ───────────────────────────
    print("\n[3/5] Testing Local ML Tool (FAISS)")
    tool_name = "search_vault"
    tool_params = {"query": "What is Python?"}

    print(f"  {C_SL}Stateless search:{C_RES}")
    resp, lat = send_tool_call(STATELESS_PORT, tool_name, tool_params, "demo-1")
    print(f"    {C_SL}Result:  {resp['result']['retrieved_context'][:80]}...{C_RES}")
    print(f"    {C_SL}Latency: {lat:.1f} ms{C_RES}")

    print(f"  {C_SF}Stateful search:{C_RES}")
    resp, lat = send_tool_call(STATEFUL_PORT, tool_name, tool_params, "demo-2")
    print(f"    {C_SF}Result:  {resp['result']['retrieved_context'][:80]}...{C_RES}")
    print(f"    {C_SF}Latency: {lat:.1f} ms{C_RES}")

    print("\n[3.5] Testing Network I/O Tool (Web Scraper)")
    tool_name = "fetch_webpage"
    tool_params = {"url": "https://news.ycombinator.com"}

    print(f"  {C_SL}Stateless scrape:{C_RES}")
    resp, lat = send_tool_call(STATELESS_PORT, tool_name, tool_params, "demo-3")
    raw_content = resp['result']['content']
    print(f"    {C_SL}Result:  {raw_content[:80].replace(chr(10), ' ')}...{C_RES}")
    print(f"    {C_SL}Latency: {lat:.1f} ms{C_RES}")

    print(f"  {C_SL}Summarizing with local LLM (gemma3:270m)...{C_RES}")
    summary = summarize_with_llm(raw_content)
    print(f"    {C_SL}Summary: {summary}{C_RES}")

    print(f"  {C_SF}Stateful scrape:{C_RES}")
    resp, lat = send_tool_call(STATEFUL_PORT, tool_name, tool_params, "demo-4")
    raw_content = resp['result']['content']
    print(f"    {C_SF}Result:  {raw_content[:80].replace(chr(10), ' ')}...{C_RES}")
    print(f"    {C_SF}Latency: {lat:.1f} ms{C_RES}")

    print(f"  {C_SF}Summarizing with local LLM (gemma3:270m)...{C_RES}")
    summary = summarize_with_llm(raw_content)
    print(f"    {C_SF}Summary: {summary}{C_RES}")

    # ── Step 4: Latency benchmark ─────────────────────────
    ROUNDS = 5
    print(f"\n[4/5] Latency benchmark (FAISS vs Network Scraper)")

    # WARMUP phase to eliminate cold-start bias on the SentenceTransformer
    print("  Warming up ML models (1 call each)...")
    send_tool_call(STATELESS_PORT, "search_vault", {"query": "warmup"}, "warmup-1")
    send_tool_call(STATEFUL_PORT, "search_vault", {"query": "warmup"}, "warmup-2")

    print(f"\n  -- Benchmark 1: FAISS Search (Local ML/CPU Bound) --")
    tool_name = "search_vault"
    tool_params = {"query": "Machine learning"}
    
    print(f"  {C_SL}Stateless server:{C_RES}")
    faiss_sl_lats = benchmark_latency(STATELESS_PORT, "stateless", tool_name, tool_params, ROUNDS, C_SL)
    
    print(f"  {C_SF}Stateful server (writes to Redis):{C_RES}")
    faiss_sf_lats = benchmark_latency(STATEFUL_PORT, "stateful", tool_name, tool_params, ROUNDS, C_SF)

    print(f"\n  -- Benchmark 2: Web Scraper (Network I/O Bound) --")
    tool_name = "fetch_webpage"
    tool_params = {"url": "https://example.com"}
    
    print(f"  {C_SL}Stateless server:{C_RES}")
    net_sl_lats = benchmark_latency(STATELESS_PORT, "stateless", tool_name, tool_params, ROUNDS, C_SL)
    
    print(f"  {C_SF}Stateful server (writes to Redis):{C_RES}")
    net_sf_lats = benchmark_latency(STATEFUL_PORT, "stateful", tool_name, tool_params, ROUNDS, C_SF)


    # ── Summary ───────────────────────────────────────────
    avg_faiss_sl = sum(faiss_sl_lats) / len(faiss_sl_lats)
    avg_faiss_sf = sum(faiss_sf_lats) / len(faiss_sf_lats)
    
    avg_net_sl = sum(net_sl_lats) / len(net_sl_lats)
    avg_net_sf = sum(net_sf_lats) / len(net_sf_lats)

    redis_bloat_kb = session_store.total_bloat_kb

    print("\n" + "=" * 60)
    print("STATISTICS: LOCAL ML vs NETWORK I/O")
    print("=" * 60)
    print(f"  Local ML FAISS Search (avg {ROUNDS} rounds):")
    print(f"    {C_SL}Stateless (No cache):    {avg_faiss_sl:.1f} ms{C_RES}")
    print(f"    {C_SF}Stateful  (Redis Write): {avg_faiss_sf:.1f} ms{C_RES}")
    print(f"      -> Difference: {abs(avg_faiss_sl - avg_faiss_sf):.2f} ms")
    
    print(f"\n  Network Web Scraper (avg {ROUNDS} rounds):")
    print(f"    {C_SL}Stateless (No cache):    {avg_net_sl:.1f} ms{C_RES}")
    print(f"    {C_SF}Stateful  (Redis Write): {avg_net_sf:.1f} ms{C_RES}")
    print(f"      -> Difference: {abs(avg_net_sl - avg_net_sf):.2f} ms")
    
    print(f"\n  Redis session bloat: {redis_bloat_kb:.1f} KB (total across DB)")
    print("=" * 60)


if __name__ == "__main__":
    main()