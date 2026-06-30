# Local Heterogeneous Swarm: Cooperative vs Adversarial

This project demonstrates a multi-agent AI swarm running completely locally using Ollama. It features a dynamic workflow where an Orchestrator model routes tasks between a Cooperative agent (writing code) and an Adversarial agent (finding flaws) until the code is robust.

## Architecture

- **Orchestrator:** `phi3:mini` - Dynamically decides who speaks next based on conversation history.
- **Cooperative Agent:** `qwen3.5:4b` - Drafts and refines code implementations.
- **Adversarial Agent:** `deepseek-coder:6.7b` - Critiques code, finding bugs and edge cases.
- **Vector Memory:** `FAISS` with `SentenceTransformers` (`all-MiniLM-L6-v2`) for instant context retrieval.
- **Caching:** `Redis` to instantly recall identical prompts and save compute.

## Setup

1. **Install Python Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Redis Server:**
   You must have Redis running locally (e.g. via Docker) for the caching to work, although the script has a graceful fallback if Redis is unavailable.
   ```bash
   docker run -d --name redis-swarm -p 6379:6379 redis
   ```

3. **Install Ollama & Download Models:**
   Ensure [Ollama](https://ollama.com) is installed and the required models are pulled:
   ```bash
   ollama pull phi3:mini
   ollama pull qwen3.5:4b
   ollama pull deepseek-coder:6.7b
   ```

## Usage

Run the demo script:
```bash
python demo.py
```

The script will automatically set up the FAISS index, initialize the LLM clients, and start the swarm debate over a notoriously tricky technical interview question: implementing an LRU Cache!
