# 100% Local Arctic Swarm Architecture

This project is a 100% local replica of the massive multi-agent AI "Arctic Swarm" architecture. Instead of relying on expensive cloud LLMs, managed vector databases, and cloud Pub/Sub, this project proves you can run an enterprise-grade asynchronous research swarm directly on your machine.

## Architecture

1. **Redis (Message Broker)**: Replaces cloud Pub/Sub queues. Acts as a Gated Bulletin Board System connecting the decoupled agents.
2. **Ollama**: Replaces cloud LLMs. We use `qwen3.5:4b` for ultra-fast routing/query optimization, and `gemma4:26b` for heavy deep research synthesis.
3. **FAISS & SentenceTransformers**: Replaces managed vector databases. We use `all-MiniLM-L6-v2` (384 dimensions) for instant, local document retrieval.

## The Agents (Threads)
- **GovernanceAgent** (`qwen3.5:4b`): Analyzes raw incoming queries and routes them to the correct governance queue (consensus, delegated, meritocratic).
- **CoordinatorAgent** (`qwen3.5:4b`): Acts as a Query Planner, optimizing the raw query into a concise list of search keywords for the vector DB.
- **VectorStorageAgent**: Embeds the keywords and retrieves the most relevant context chunks from the local FAISS index.
- **ResearchAgent** (`gemma4:26b`): Reads the query + context and streams a highly detailed, synthesized response.

## Setup

1. **Install Requirements**:
   It is highly recommended to use a virtual environment (e.g. `uv` or `venv`).
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Redis**:
   You must have Redis running locally on port 6379. Using Docker is the easiest way:
   ```bash
   docker run -d -p 6379:6379 --name redis_local redis
   ```

3. **Install Ollama & Pull Models**:
   Ensure you have [Ollama](https://ollama.com/) installed and running. Pull the required models:
   ```bash
   ollama pull qwen3.5:4b
   ollama pull gemma4:26b
   ```

## Usage

Run the demo script:
```bash
python demo_swarm.py
```

The script will:
- Connect to Redis.
- Boot up all 4 agent threads.
- Embed the realistic mock documents into FAISS (the first run will take a moment to download the `all-MiniLM-L6-v2` weights).
- Submit a test query and stream the final answer back to the terminal with ANSI color coding for each agent's logs.
