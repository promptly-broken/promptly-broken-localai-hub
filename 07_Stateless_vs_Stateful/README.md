# Stateless vs Stateful MCP Server Benchmark

This repository demonstrates the practical differences in latency and memory overhead between a **Stateless** Model Context Protocol (MCP) server and a traditional **Stateful** MCP server backed by a Redis session store.

## The Experiment
We benchmark both architectures using two realistic Agentic workflows:
1. **Local ML Vector Search (FAISS)** - A heavily CPU-bound task that takes ~7ms.
2. **Web Scraper & LLM Summarization** - A heavily Network and GPU I/O-bound task that scrapes a URL and summarizes the text using a local Ollama model (`gemma3:270m`).

The goal is to show that when an agent touches the network or performs an LLM inference, the tiny 1ms latency overhead of maintaining state in Redis becomes mathematically invisible.

## Prerequisites

1. **Python 3.10+**
2. **Redis**: You must have a local Redis server running on port `6379`.
   ```bash
   docker run -d -p 6379:6379 redis
   ```
3. **Ollama**: You must have Ollama running locally with the `gemma3:270m` model installed for the LLM summarization step.
   ```bash
   ollama run gemma3:270m
   ```

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Benchmark

Simply run the script:
```bash
python demo.py
```

The script will:
1. Create a dummy `obsidian_vault` directory with markdown files.
2. Start both the Stateless (Port `8100`) and Stateful (Port `8101`) FastAPI servers.
3. Perform a quick smoke test for both tools.
4. Run a benchmark comparing the latency between the stateless and stateful endpoints.
5. Print beautifully colorized results to your terminal.
