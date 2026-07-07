# Local Agentic Workflow: Hybrid Routing

This directory contains the code to demonstrate a highly-scalable, robust **Hybrid Routing Pipeline** for Agentic AI workflows with 100+ tools.

## The Problem
When you have an "enterprise-scale" toolkit (e.g. 100+ internal APIs), you cannot provide all 100 tool definitions to your LLM without destroying its ability to reason logically about dependencies and conditionals. 

Relying strictly on Vector Search (RAG) to fetch tools fails because Vector Search relies entirely on semantic/keyword overlap and cannot evaluate logical conditions (e.g. *"If X happens, do Y"*).

## The Solution
This demo proves the necessity of a 3-stage pipeline:
1. **Coarse Filter (Vector Search):** Uses SentenceTransformers to quickly whittle down 100+ tools into a Top 20 semantic list.
2. **Fine Filter (Router LLM):** A fast local model (`qwen3.5:4b`) reads those 20 tool descriptions, understands the implicit dependencies and conditionals in the prompt, and extracts exactly the 5-7 required tools.
3. **Execution (Main LLM):** A capable local model (`qwen3-coder:30b`) effortlessly sequences and executes the prompt with a pristine tool context.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure you have Ollama installed and the required models pulled:
   ```bash
   ollama run qwen3.5:4b
   ollama run qwen3-coder:30b
   ```

## Running the Demo

Execute the main script:
```bash
python main.py
```

The terminal will live-stream the difference between what a pure Vector Search approach retrieves vs. what the Hybrid LLM Router retrieves, proving why the routing layer is strictly necessary.
