# 01 - Local Governor

This demo shows how to use a two-layer governor pattern to monitor and validate interactions with a large language model (LLM) agent, entirely locally.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure Ollama is running with the required models (`qwen3-coder:30b` and `qwen3.5:4b` or equivalents in the code).

4. Run the demo:
   ```bash
   python main.py
   ```
