# Local LLM Wiki

A fully local, agentic knowledge base that runs entirely on your laptop. No cloud APIs, no privacy risks—just pure autonomous AI.

Inspired by Andrej Karpathy's vision for the "LLM OS" and agentic wikis, this project replaces cloud dependencies with a local Python agent.

## Architecture
- **Ollama**: Runs powerful local models (e.g., Codestral, Llama 3).
- **sentence-transformers**: Builds vector embeddings locally.
- **FAISS**: Lightning-fast semantic search with cosine-similarity matching.
- **Redis**: Caches LLM responses to prevent duplicate work.
- **Watchdog**: Monitors the Obsidian vault for manual edits and triggers automatic re-indexing.
- **Obsidian**: Markdown file editor and vault.

## Setup
1. Ensure [Ollama](https://ollama.com/) is installed and running.
2. Pull a model (e.g., `ollama pull codestral`).
3. Make sure you have a Redis server running (e.g., `brew install redis` or `docker run -p 6379:6379 -d redis`).
4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main script to start the agent and simulate user queries:
```bash
python main.py
```
The agent will create an `obsidian_vault` directory, index the markdown files, and answer queries (saving the output as new notes automatically).
