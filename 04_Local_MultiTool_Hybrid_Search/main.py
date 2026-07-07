import ollama
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
import json
import re
from rich.console import Console
from rich.table import Table

console = Console()

# Configure the LLM Router Model
print("Configuring Models...")
ROUTER_MODEL = 'qwen3.5:4b' # Fast local model for routing
MAIN_MODEL = 'qwen3-coder:30b' # Main execution model

# ==========================================
# 1. Define a large pool of tools (The Noise)
# ==========================================
# 20 Base tools
base_tools = [
    {"name": "get_weather", "description": "Get the current weather for a location."},
    {"name": "read_file", "description": "Read the contents of a local file."},
    {"name": "write_file", "description": "Write data to a local file."},
    {"name": "search_github", "description": "Search GitHub repositories for code."},
    {"name": "send_slack_message", "description": "Send a message to a Slack channel."},
    {"name": "calculate_mortgage", "description": "Calculate monthly mortgage payments."},
    {"name": "translate_text", "description": "Translate text from one language to another."},
    {"name": "get_stock_price", "description": "Get the current stock price of a company."},
    {"name": "generate_image", "description": "Generate an image using AI."},
    {"name": "search_wikipedia", "description": "Search Wikipedia for general information."},
    {"name": "play_music", "description": "Play a specific song on Spotify."},
    {"name": "turn_on_lights", "description": "Turn on smart home lights."},
    {"name": "book_flight", "description": "Book a flight to a destination."},
    {"name": "order_food", "description": "Order food from a restaurant."},
    {"name": "check_email", "description": "Check the user's latest emails."},
    {"name": "create_calendar_event", "description": "Create a new event in the calendar."},
    {"name": "get_sports_scores", "description": "Get the latest scores for a sports team."},
    {"name": "calculate_tip", "description": "Calculate the tip for a restaurant bill."},
    {"name": "convert_currency", "description": "Convert money from one currency to another."},
    {"name": "set_alarm", "description": "Set an alarm for a specific time."}
]

# Generate 80 Dummy Tools to simulate a massive 100-tool enterprise environment
dummy_tools = [{"name": f"enterprise_tool_{i}", "description": f"Internal enterprise function to process data stream {i}."} for i in range(1, 81)]
tool_definitions = base_tools + dummy_tools

USER_PROMPTS = [
    "First, fetch the current stock price of AAPL and translate the resulting price summary into Spanish. Next, search Wikipedia to see who the current CEO of Apple is, and send a Slack message to the #finance channel with both the Spanish translation and the CEO's name. If it is raining in New York, turn on the smart home lights.",
    
    "Generate an image of a victorious sports team and set an alarm for 7 PM so I remember to watch the game. While I wait, check the latest sports scores for the Golden State Warriors and check my email to see if my friends have RSVP'd for the pizza party. Order food for 5 people if they did.",
    
    "I need to plan my upcoming trip. Book a flight to Paris, get the weather forecast for Paris to know what to pack, and translate 'Where is the nearest bakery' to French. Also, read my 'packing_list.txt' file, add 'umbrella' if it's raining, and create a calendar event for the flight departure."
]

# ==========================================
# 2. Vector Search (Coarse Filter)
# ==========================================
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Pre-compute embeddings for all 100 tools
tool_descriptions = [t["description"] for t in tool_definitions]
print("Embedding 100 tool descriptions...")
tool_embeddings = embedder.encode(tool_descriptions)

def vector_search(prompt, tools, top_n=20):
    prompt_embedding = embedder.encode(prompt)
    
    # Calculate Cosine Similarities
    similarities = []
    for i, tool_emb in enumerate(tool_embeddings):
        sim = np.dot(prompt_embedding, tool_emb) / (norm(prompt_embedding) * norm(tool_emb))
        similarities.append((i, sim))
    
    # Sort by descending similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    top_indices = [idx for idx, sim in similarities[:top_n]]
    return [tools[i] for i in top_indices]

# ==========================================
# 3. LLM Routing Layer (Fine Filter)
# ==========================================
def route_top_tools(prompt, tools, top_k=7):
    tool_descriptions_str = "\n".join([f"- {t['name']}: {t['description']}" for t in tools])
    
    router_prompt = f"""
You are an intelligent router that selects the necessary tools to solve a user's prompt. 
You must think logically about conditional statements and dependencies.

User Prompt: {prompt}

Available Tools:
{tool_descriptions_str}

List ONLY the names of the tools needed, formatted as a valid JSON array of strings. Do not output anything else.
Example: ["tool1", "tool2"]
"""
    
    response = ollama.chat(
        model=ROUTER_MODEL,
        messages=[{'role': 'user', 'content': router_prompt}],
        options={'temperature': 0.0}
    )
    
    raw_content = response['message']['content']
    
    try:
        match = re.search(r'\[.*?\]', raw_content, re.DOTALL)
        if match:
            selected_names = json.loads(match.group(0))
        else:
            selected_names = []
    except Exception as e:
        selected_names = []
        
    selected_tools = [t for name in selected_names for t in tools if t['name'] == name]
    return selected_tools[:top_k]

# ==========================================
# 4. Agent Execution (Before & After)
# ==========================================
def run_agent(prompt, tools_to_inject):
    formatted_tools = [{
        'type': 'function',
        'function': {
            'name': t['name'],
            'description': t['description'],
            'parameters': {'type': 'object', 'properties': {'location': {'type': 'string'}}}
        }
    } for t in tools_to_inject]

    response = ollama.chat(
        model=MAIN_MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        tools=formatted_tools
    )
    
    if response.get('message', {}).get('tool_calls'):
        calls = [tc['function']['name'] for tc in response['message']['tool_calls']]
    else:
        calls = []
        
    return calls

if __name__ == "__main__":
    print(f"\nMain Model: {MAIN_MODEL}")
    
    summary_data = []
    
    for i, prompt in enumerate(USER_PROMPTS, 1):
        print(f"\n==========================================")
        print(f"Test Case {i}")
        print(f"Goal: {prompt}")
        print(f"==========================================")
        
        # 1. Vector Search Only: Grab top 7 directly to the agent
        # The vector search looks for keyword overlap, ignoring logical flow.
        vector_only_tools = vector_search(prompt, tool_definitions, top_n=7)
        print(f"\n=== Running Vector Search Only (Top 7) ===")
        print(f"Tools passed to agent: {[t['name'] for t in vector_only_tools]}")
        vector_only_calls = run_agent(prompt, vector_only_tools)
        print(f"\033[91mAgent executed: {vector_only_calls}\033[0m")
        
        # 2. Hybrid Search: Vector (Top 20) -> Router (Top 7) -> Agent
        # The router reads the Top 20 and logically selects the true dependencies.
        hybrid_vector_tools = vector_search(prompt, tool_definitions, top_n=20)
        print(f"\n--- LLM Routing Layer Executing ---")
        hybrid_router_tools = route_top_tools(prompt, hybrid_vector_tools, top_k=7)
        router_selected_names = [t['name'] for t in hybrid_router_tools]
        print(f"Tools selected by LLM router: {router_selected_names}")
        
        print(f"\n=== Running Hybrid Approach (Top 7) ===")
        hybrid_calls = run_agent(prompt, hybrid_router_tools)
        print(f"\033[92mAgent executed: {hybrid_calls}\033[0m")
        
        summary_data.append((prompt, vector_only_calls, router_selected_names))
        
    # Print Summary Table with Rich
    table = Table(title="100-Tool Pool: Vector Search vs Hybrid Pipeline", show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("Query", style="cyan", no_wrap=False)
    table.add_column("Vector Search Only (Top 7)", style="red", no_wrap=False)
    table.add_column("Hybrid (Vector Top 20 -> LLM Router)", style="green", no_wrap=False)

    for q, vector, hybrid in summary_data:
        v_str = "\n".join(vector) if vector else "None"
        h_str = "\n".join(hybrid) if hybrid else "None"
        table.add_row(q, v_str, h_str)
        
    console.print("\n\n")
    console.print(table)
    console.print("\n")
