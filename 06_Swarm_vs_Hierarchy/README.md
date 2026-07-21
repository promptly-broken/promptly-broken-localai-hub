# Swarm vs Hierarchy

A Demo of Multi-Agent Architectures using Local LLMs.

This directory explores the differences between Swarm and Hierarchy agent architectures by implementing both patterns to solve a research and writing task.

## Features
- **Two Architectures**: Implements a top-down Manager-Worker hierarchy and a peer-to-peer autonomous Swarm.
- **Local LLMs**: Configured to run against local models (e.g., via Ollama). Default models include `gemma4:26b`, `phi3:mini`, and `gemma4:12b-mlx`.
- **Live Tool Calling & Streaming**: Features an ArXiv search tool and streams agent thoughts to the console.
- **Observability**: Generates Mermaid sequence diagrams (`hierarchy_flow.md` and `swarm_flow.md`) and logs state to `execution_state.jsonl`.

## Usage
Ensure the required packages are installed, then run:
```bash
python swarm_vs_hierarchy.py
```
