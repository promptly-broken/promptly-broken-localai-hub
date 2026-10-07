"""
Promptly Broken — MCTS + Process Reward Model Interactive Visualizer
Streamlit UI to inspect real-time tree search, step evaluations, and pruning.
"""

import sys
import time
from pathlib import Path
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcts.search import MCTSSearchEngine
from mcts.verifier import PRMVerifier
from generate_prm_dataset_v2 import download_or_load_csv

st.set_page_config(
    page_title="Promptly Broken | MCTS + PRM Engine",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1a1c24;
        border: 1px solid #2e3440;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .badge-win {
        background-color: #2e7d32;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-pruned {
        background-color: #c62828;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-target {
        background-color: #f57f17;
        color: black;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Promptly Broken: Test-Time Compute Scaling Engine")
st.markdown("**Monte Carlo Tree Search (MCTS) + Process Reward Model (PRM) on Apple Silicon**")
st.caption("Overcoming greedy LLM hallucination through deliberate step-level verification and backtracking.")

# Sidebar Configuration
st.sidebar.header("⚙️ Search Hyperparameters")

puzzles = download_or_load_csv()
preset_options = {
    "Custom Numbers": None,
    "The Deceptive Fraction Trap [3, 3, 8, 8]": [3, 3, 8, 8],
    "Classic Permutation [4, 4, 6, 8]": [4, 4, 6, 8],
    "Rank 901 Benchmark [4, 5, 6, 10]": [4, 5, 6, 10],
    "Rank 902 Benchmark [1, 2, 4, 7]": [1, 2, 4, 7],
    "Rank 908 Hard [2, 3, 6, 9]": [2, 3, 6, 9],
    "Rank 909 Hard [1, 3, 5, 9]": [1, 3, 5, 9],
}

selected_preset = st.sidebar.selectbox("Select Benchmark Puzzle:", list(preset_options.keys()))

if selected_preset == "Custom Numbers":
    custom_input = st.sidebar.text_input("Enter 4 integers separated by space:", "3 3 8 8")
    try:
        puzzle_numbers = [int(x) for x in custom_input.strip().split()][:4]
    except ValueError:
        puzzle_numbers = [3, 3, 8, 8]
else:
    puzzle_numbers = preset_options[selected_preset]

verifier_mode = st.sidebar.selectbox(
    "PRM Verifier Engine:",
    ["symbolic", "ollama"],
    format_func=lambda x: "Exact Symbolic Oracle (Sub-millisecond)" if x == "symbolic" else "Local LLM Verifier (Ollama Qwen2.5-Coder)"
)

max_sims = st.sidebar.slider("Max Tree Simulations:", min_value=20, max_value=300, value=120, step=10)
k_branching = st.sidebar.slider("Branching Factor (K):", min_value=2, max_value=6, value=4)
c_puct = st.sidebar.slider("PUCT Exploration (c_puct):", min_value=0.5, max_value=3.0, value=1.414, step=0.1)

run_button = st.button("🚀 Run MCTS Search", type="primary")


def render_tree_node(node_dict: dict, depth: int = 0):
    """Recursively render expandable tree nodes in Streamlit."""
    action = node_dict["action"]
    state = node_dict["state"]
    is_pruned = node_dict["is_pruned"]
    is_target = node_dict["is_target"]
    prm_score = node_dict["prm_score"]
    critique = node_dict["critique"]
    visits = node_dict["visits"]
    q_val = node_dict["q_value"]
    children = node_dict["children"]

    if is_target:
        badge = "🏆 TARGET 24 REACHED"
        color = "#f57f17"
    elif is_pruned:
        badge = "❌ PRUNED DEAD-END"
        color = "#c62828"
    else:
        badge = "✅ SOUND STEP"
        color = "#2e7d32"

    label = f"{'  ' * depth}▸ **{action}** | Remaining: `{state}` | [{badge}] (Visits: {visits}, Q: {q_val})"

    with st.expander(label, expanded=(depth < 1 or is_target)):
        st.write(f"**Action:** `{action}`")
        st.write(f"**State Pool:** `{state}`")
        st.write(f"**PRM Score:** `{prm_score}`")
        if critique:
            st.info(f"**PRM Critique:** {critique}")
        st.write(f"**Search Stats:** Visited {visits} times | Estimated Q-value: `{q_val}`")

        if children:
            st.markdown(f"**Children ({len(children)} branches):**")
            for child in children:
                render_tree_node(child, depth + 1)


if run_button or "mcts_res" in st.session_state:
    if run_button:
        engine = MCTSSearchEngine(
            verifier=PRMVerifier(mode=verifier_mode),
            c_puct=c_puct,
            max_simulations=max_sims,
            k_branching=k_branching
        )
        with st.spinner("Searching state tree and evaluating branches with PRM..."):
            res = engine.solve(puzzle_numbers)
            st.session_state["mcts_res"] = res
            st.session_state["puzzle"] = puzzle_numbers

    res = st.session_state["mcts_res"]
    puzzle = st.session_state["puzzle"]

    st.divider()

    # Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Outcome", "SOLVED" if res.success else "FAILED", delta="100% Verified" if res.success else None)
    with c2:
        st.metric("Latency", f"{res.execution_time_sec * 1000:.1f} ms")
    with c3:
        st.metric("Nodes Explored", str(res.total_nodes))
    with c4:
        st.metric("Dead Ends Pruned", str(res.pruned_nodes))
    with c5:
        pruned_ratio = (res.pruned_nodes / max(1, res.total_nodes)) * 100
        st.metric("Pruning Efficiency", f"{pruned_ratio:.1f}%")

    st.subheader(f"🎯 Puzzle Input: `{puzzle}`")

    if res.success:
        st.success(f"**Solution Discovered in {res.simulations_run} Tree Simulations:**")
        for i, step in enumerate(res.solution_steps, 1):
            st.markdown(f"**Step {i}:** `{step}`")
    else:
        st.error(f"**Search Result:** {res.verification_message}")

    st.divider()
    st.subheader("🌲 Interactive Search Tree & PRM Pruning Inspector")
    st.caption("Inspect each branch. Notice how the PRM assigns 0.0 to dead-end traps, allowing the engine to backtrack early.")

    tree_dict = res.root.to_dict()
    render_tree_node(tree_dict)
