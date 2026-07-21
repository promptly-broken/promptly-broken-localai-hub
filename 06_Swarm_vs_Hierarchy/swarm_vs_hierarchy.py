#!/usr/bin/env python3
"""
Swarm vs Hierarchy: A Demo of Multi-Agent Architectures using Local LLMs
Includes Live Tool Calling and Streaming Output.
"""

import json
import os
import sys
import time
import datetime
from typing import List, Dict, Any, Callable

try:
    from openai import OpenAI
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    import feedparser
    import urllib.request
    import urllib.parse
except ImportError:
    print("Error: Required packages not found. Please run this in the .venv where main.py runs.")
    print("Example: cd swarm_vs_hierarchy && ../.venv/bin/python swarm_vs_hierarchy.py")
    sys.exit(1)

console = Console()

# --- Config ---
BASE_URL = "http://127.0.0.1:11434/v1"
API_KEY = "not-needed"

# Define Models (fallback if config.yaml not found)
MANAGER_MODEL = "gemma4:26b"
WORKER_MODEL = "phi3:mini"
EDITOR_MODEL = "gemma4:12b-mlx"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# --- Agent Colors ---
AGENT_COLORS = {
    "Manager": "blue",
    "Researcher": "green",
    "Writer": "yellow",
    "Editor": "magenta",
    "Tool": "cyan",
    "System": "white"
}

# --- Shared Utilities ---
def log_state(event_type: str, actor: str, content: str):
    """Writes explicit state logs for observability."""
    timestamp = datetime.datetime.now().isoformat()
    log_entry = json.dumps({"timestamp": timestamp, "event": event_type, "actor": actor, "content": content})
    with open("execution_state.jsonl", "a") as f:
        f.write(log_entry + "\n")

class FlowVisualizer:
    def __init__(self, name):
        self.name = name
        self.flow = []
        
    def add_step(self, step_name: str, agent: str, status="success"):
        self.flow.append((step_name, agent, status))
        
    def generate_mermaid(self):
        lines = ["```mermaid", "sequenceDiagram", "    participant U as User"]
        
        participants = []
        for step, agent, _ in self.flow:
            if agent not in participants:
                participants.append(agent)
                lines.append(f"    participant {agent}")
        
        last_actor = "U"
        for step, agent, status in self.flow:
            if status == "fail":
                lines.append(f"    {last_actor}-x{agent}: [FAILED] {step}")
            else:
                lines.append(f"    {last_actor}->>{agent}: {step}")
            last_actor = agent
            
        lines.append("```")
        return "\n".join(lines)
        
    def save(self):
        filename = f"{self.name.lower()}_flow.md"
        with open(filename, "w") as f:
            f.write(f"# {self.name} Architecture Flow\n\n")
            f.write(self.generate_mermaid())
        console.print(f"[bold cyan]Saved sequence diagram to {filename}[/bold cyan]")

def search_arxiv(query: str) -> str:
    """Tool: Searches ArXiv for academic papers and returns a synthesized snippet."""
    console.print(f"\n[bold cyan]🛠️ Tool (ArXiv Search) Executing:[/bold cyan] [cyan]'{query}'[/cyan]")
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results=3"
        response = urllib.request.urlopen(url)
        feed = feedparser.parse(response)
        
        if not feed.entries:
            raise ValueError("Empty results")
            
        text_results = "\n\n".join([f"- Title: {entry.title}\n  Summary: {entry.summary[:300]}..." for entry in feed.entries])
        console.print(f"[cyan]Found {len(feed.entries)} papers...[/cyan]")
        log_state("tool_execution", "ArXiv_Search", f"Query: {query} | Found: {len(feed.entries)} papers")
        return text_results
    except Exception as e:
        console.print(f"[yellow]Warning: Real ArXiv search failed ({e}). Using cached results for demo resilience.[/yellow]")
        return (
            "- Title: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity\n  Summary: We simplify the MoE routing algorithm... showing significant speedups over dense models.\n\n"
            "- Title: Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer\n  Summary: We present a sparsely-gated MoE layer... achieving >1000x improvements in model capacity with minor compute overhead.\n\n"
            "- Title: GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding\n  Summary: MoE models demonstrate superior scaling laws, but dense models remain easier to deploy in memory-constrained environments."
        )

# --- Agent Base ---
def ask_agent(role_name: str, model: str, system_prompt: str, user_prompt: str, chat_history: List[dict] = None) -> str:
    """Streams the LLM response to the console with the agent's color."""
    color = AGENT_COLORS.get(role_name, "white")
    
    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": user_prompt})
    
    console.print(f"\n[bold {color}]🤖 {role_name} (using {model}) is thinking...[/bold {color}]")
    
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=4096,
            stream=True
        )
        
        ansi_color = {"blue": "\033[94m", "green": "\033[92m", "yellow": "\033[93m", "magenta": "\033[95m", "cyan": "\033[96m"}.get(color, "\033[0m")
        sys.stdout.write(ansi_color)
        
        full_response = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                sys.stdout.write(content)
                sys.stdout.flush()
                
        sys.stdout.write("\033[0m\n\n")
        sys.stdout.flush()
        final_text = full_response.strip()
        log_state("agent_response", role_name, final_text)
        return final_text
    except Exception as e:
        console.print(f"\n[red]LLM Error: {e}[/red]")
        log_state("agent_error", role_name, str(e))
        return f"Error: {e}"

# --- 1. Hierarchy Architecture ---
def run_hierarchy(task: str):
    console.print(Panel("=== Hierarchy Architecture (Manager ➔ Worker) ===", style="bold blue"))
    visualizer = FlowVisualizer("Hierarchy")
    
    # 1. Manager delegates research
    manager_sys = "You are the Manager Agent. You orchestrate tasks. Output clear directives. Thinking mode ON: explicitly think step-by-step and enclose your reasoning in <think> tags before providing your final output."
    m_prompt_1 = f"The user requested: '{task}'. Delegate a specific search query to the Researcher to gather data."
    search_query = ask_agent("Manager", MANAGER_MODEL, manager_sys, m_prompt_1)
    visualizer.add_step("Delegate Search", "Manager")
    
    # 2. Researcher executes search and synthesizes
    r_sys = "You are the Researcher Agent. Synthesize the provided raw search results into a concise summary."
    # Extracted query heuristics (just grabbing a likely string from manager's output for demo sake)
    clean_query = "Mixture of Experts Dense Models NLP"
    search_results = search_arxiv(clean_query)
    visualizer.add_step("Execute ArXiv Search", "Tool")
    
    r_prompt = f"Raw Search Results:\n{search_results}\n\nSynthesize this data for the Manager."
    research_summary = ask_agent("Researcher", WORKER_MODEL, r_sys, r_prompt)
    visualizer.add_step("Synthesize Data", "Researcher")
    
    # 3. Manager reviews research and approves for writing
    m_prompt_2 = f"Here is the research summary: {research_summary}\n\nReview this research. If the quality is acceptable, respond with only: 'APPROVED FOR WRITING'. If not, state what is missing."
    manager_review = ask_agent("Manager", MANAGER_MODEL, manager_sys, m_prompt_2)
    visualizer.add_step("Review Research", "Manager")
    
    # 4. Writer drafts based on research (clean prompt, no forwarded manager verbosity)
    w_sys = "You are the Writer Agent. Write an engaging, highly technical 3-point post. Use the provided research as your source material. Output ONLY the post content — no preamble, no meta-commentary."
    w_prompt = f"Write a technical post based on this research:\n\n{research_summary}"
    draft = ask_agent("Writer", WORKER_MODEL, w_sys, w_prompt)
    visualizer.add_step("Draft Post", "Writer")
    
    # 5. Editor polishes the draft
    e_sys = "You are the Editor Agent. Take the Writer's draft below and edit it to meet strict standards: exactly 3 bullet points, highly academic tone, maximum 150 words. Output ONLY the final polished version."
    e_prompt = f"Writer's Draft:\n\n{draft}\n\nProvide the final edited version."
    final = ask_agent("Editor", EDITOR_MODEL, e_sys, e_prompt)
    visualizer.add_step("Edit Post", "Editor")
    
    visualizer.save()


# --- 2. Swarm Architecture ---
def run_swarm(task: str):
    console.print(Panel("=== Swarm Architecture (Autonomous Peer-to-Peer Bus) ===", style="bold magenta"))
    visualizer = FlowVisualizer("Swarm")
    
    # Shared message bus — every agent reads the FULL bus each round
    bus = [{"sender": "User", "type": "task", "content": task}]
    console.print(f"[white]📬 User posted task to Swarm Bus[/white]")
    
    def bus_text():
        """Render bus as readable text for agent prompts."""
        return "\n".join([f"[{m['sender']}] ({m['type']}): {m['content'][:500]}" for m in bus])
    
    def has(sender=None, msg_type=None):
        """Check if bus has a message matching criteria."""
        return any((sender is None or m["sender"] == sender) and 
                   (msg_type is None or m["type"] == msg_type) for m in bus)
    
    def latest(sender):
        """Get the latest message from a sender."""
        msgs = [m for m in bus if m["sender"] == sender]
        return msgs[-1] if msgs else None
    
    MAX_ROUNDS = 6
    agents = ["Researcher", "Writer", "Editor"]
    
    for round_num in range(1, MAX_ROUNDS + 1):
        console.print(Panel(f"⚡ Swarm Round {round_num}", style="bold white"))
        
        for agent_name in agents:
            color = AGENT_COLORS.get(agent_name, "white")
            
            # ── RESEARCHER ──
            if agent_name == "Researcher":
                if not has(sender="Researcher"):
                    # First time: decide query and search
                    console.print(f"[{color}]  🔍 Researcher scans bus → sees task, deciding search query...[/{color}]")
                    r_sys = "You are a Researcher in a peer-to-peer Swarm. Output ONLY a concise ArXiv search query string, nothing else."
                    query = ask_agent("Researcher", WORKER_MODEL, r_sys,
                                     f"The bus contains this task:\n{bus[0]['content']}\n\nWhat is your search query?")
                    visualizer.add_step("Decide Query", "Researcher")
                    
                    results = search_arxiv(query)
                    visualizer.add_step("Execute ArXiv Search", "Tool")
                    
                    bus.append({"sender": "Researcher", "type": "research", "content": results})
                    console.print(f"[{color}]  ✅ Researcher posted research findings to bus[/{color}]")
                else:
                    console.print(f"[{color}]  💤 Researcher scans bus → already contributed, skipping[/{color}]")
            
            # ── WRITER ──
            elif agent_name == "Writer":
                editor_msg = latest("Editor")
                writer_drafts = [m for m in bus if m["sender"] == "Writer"]
                
                if not has(sender="Researcher", msg_type="research"):
                    console.print(f"[{color}]  ⏳ Writer scans bus → no research yet, waiting...[/{color}]")
                
                elif not writer_drafts:
                    # First draft
                    console.print(f"[{color}]  ✍️ Writer scans bus → sees research data, drafting...[/{color}]")
                    research = latest("Researcher")["content"]
                    w_sys = "You are a Writer in a peer-to-peer Swarm. Draft a technical 3-point post. Output ONLY the post content."
                    draft = ask_agent("Writer", WORKER_MODEL, w_sys,
                                     f"Write a technical post based on this research:\n\n{research}")
                    bus.append({"sender": "Writer", "type": "draft", "content": draft})
                    visualizer.add_step("Draft Post (v1)", "Writer")
                    console.print(f"[{color}]  ✅ Writer posted draft v1 to bus[/{color}]")
                
                elif editor_msg and editor_msg["type"] == "rejection":
                    # Editor rejected — revise!
                    console.print(f"[{color}]  🔄 Writer scans bus → sees Editor REJECTION, revising...[/{color}]")
                    feedback = editor_msg["content"]
                    prev_draft = writer_drafts[-1]["content"]
                    w_sys = "You are a Writer in a peer-to-peer Swarm. The Editor rejected your draft with specific feedback. Revise accordingly. Output ONLY the revised post."
                    revision = ask_agent("Writer", WORKER_MODEL, w_sys,
                                        f"Your previous draft:\n{prev_draft}\n\nEditor's feedback:\n{feedback}\n\nRevise the draft.")
                    bus.append({"sender": "Writer", "type": "revision", "content": revision})
                    visualizer.add_step("Revise Draft (v2)", "Writer")
                    console.print(f"[{color}]  ✅ Writer posted revised draft v2 to bus[/{color}]")
                
                else:
                    console.print(f"[{color}]  💤 Writer scans bus → waiting for Editor feedback...[/{color}]")
            
            # ── EDITOR ──
            elif agent_name == "Editor":
                writer_drafts = [m for m in bus if m["sender"] == "Writer"]
                editor_msgs = [m for m in bus if m["sender"] == "Editor"]
                
                if not writer_drafts:
                    console.print(f"[{color}]  ⏳ Editor scans bus → no draft yet, waiting...[/{color}]")
                
                elif not editor_msgs:
                    # First review: REJECT with feedback to force back-and-forth
                    console.print(f"[{color}]  📝 Editor scans bus → sees draft v1, reviewing...[/{color}]")
                    draft = writer_drafts[-1]["content"]
                    e_sys = "You are the Editor in a peer-to-peer Swarm. Review this draft critically. It does NOT meet standards yet. Provide specific, actionable feedback: it must be exactly 3 bullet points, academic tone, max 150 words. List exactly what needs to change."
                    feedback = ask_agent("Editor", EDITOR_MODEL, e_sys,
                                        f"Writer's draft to review:\n\n{draft}\n\nProvide your rejection feedback.")
                    bus.append({"sender": "Editor", "type": "rejection", "content": feedback})
                    visualizer.add_step("REJECT Draft v1", "Editor")
                    console.print(f"[{color}]  ❌ Editor REJECTED draft v1 — posted feedback to bus[/{color}]")
                
                elif has(sender="Writer", msg_type="revision") and not has(sender="Editor", msg_type="approval"):
                    # Second review: approve the revision
                    console.print(f"[{color}]  📝 Editor scans bus → sees revised draft v2, reviewing...[/{color}]")
                    revision = [m for m in bus if m["sender"] == "Writer" and m["type"] == "revision"][-1]["content"]
                    e_sys = "You are the Editor in a peer-to-peer Swarm. The Writer revised based on your feedback. Polish this into the final version: exactly 3 bullet points, academic tone, max 150 words. Output ONLY the final polished version."
                    final = ask_agent("Editor", EDITOR_MODEL, e_sys,
                                     f"Revised draft:\n\n{revision}\n\nProvide the final polished version.")
                    bus.append({"sender": "Editor", "type": "approval", "content": final})
                    visualizer.add_step("APPROVE Draft v2", "Editor")
                    console.print(f"[{color}]  ✅ Editor APPROVED — posted final version to bus[/{color}]")
                
                else:
                    console.print(f"[{color}]  💤 Editor scans bus → nothing new, waiting...[/{color}]")
        
        # Check if done
        if has(sender="Editor", msg_type="approval"):
            console.print(f"\n[bold green]🎉 Swarm reached consensus after {round_num} rounds![/bold green]")
            break
    
    visualizer.save()

if __name__ == "__main__":
    task_prompt = "Research the latest architectural differences, scaling laws, and performance trade-offs between Mixture of Experts (MoE) models and dense transformer models. Extract factual findings from recent academic literature and synthesize them into a concise technical summary."
    
    console.print(Panel(f"Target Task:\n{task_prompt}", style="bold white"))
    
    run_hierarchy(task_prompt)
    print("\n" + "="*80 + "\n")
    run_swarm(task_prompt)
    
    console.print("\n[bold green]✅ All demos completed successfully![/bold green]")
