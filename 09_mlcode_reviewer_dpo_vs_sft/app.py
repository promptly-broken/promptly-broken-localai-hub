#!/usr/bin/env python3
"""
app.py – Streamlit "Before & After" UI for the SFT-tuned PR Reviewer

Compare the raw Qwen2.5-Coder-7B base model against the
locally SFT-tuned variant, side by side.

Usage:
    streamlit run app.py
"""

import streamlit as st
import ollama
from mlx_lm import load, generate

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Senior ML PR Reviewer – SFT Demo",
    page_icon="🔬",
    layout="wide",
)

# ── Constants ───────────────────────────────────────────────
OLLAMA_BASE_MODEL = "qwen2.5-coder:32b"
FUSED_MODEL_PATH = "sft_32b_adapters"

PROMPT_TEMPLATE = """\
You are a senior Machine Learning engineer reviewing a GitHub Pull Request.
Review the following code diff from branch `{feature}` and provide detailed, actionable feedback. Focus on ML-specific correctness, not style.

{diff}"""

EXAMPLE_SNIPPET = """\
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)
print("Test accuracy:", model.score(X_test, y_test))
"""

# ── Model loading (cached) ─────────────────────────────────
@st.cache_resource(show_spinner="Loading SFT-tuned 32B model…")
def load_fused_model():
    try:
        model, tokenizer = load("mlx-community/Qwen2.5-Coder-32B-4bit", adapter_path=FUSED_MODEL_PATH)
        return model, tokenizer
    except Exception as e:
        st.error(f"Could not load tuned model from '{FUSED_MODEL_PATH}'. Did you run the fusion step? Error: {e}")
        st.stop()


def build_prompt(code: str) -> str:
    """Wrap the user's code in the exact review prompt format from training."""
    # If the user didn't paste markdown backticks, wrap it in a diff block for them
    if not code.strip().startswith("```"):
        diff = f"```diff\n{code.strip()}\n```"
    else:
        diff = code.strip()
        
    return PROMPT_TEMPLATE.format(feature="feature/review", diff=diff)


def run_mlx_inference(model, tokenizer, prompt: str, max_tokens: int) -> str:
    """Generate a review from the SFT-tuned MLX model."""
    # BASE MODELS ARE NOT CHAT MODELS! 
    # Do NOT apply chat_template, or the model will ignore the prompt
    # and just hallucinate the training data from scratch!
    formatted_prompt = prompt + "\n\n"

    response = generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    return response


def run_ollama_inference(prompt: str) -> str:
    """Generate a review from the local Ollama base model."""
    try:
        response = ollama.chat(
            model=OLLAMA_BASE_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"⚠️ **Ollama Error:** Could not connect to Ollama or model '{OLLAMA_BASE_MODEL}' not found. Make sure Ollama is running.\n\nDetails: {e}"


# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://cdn-uploads.huggingface.co/production/uploads/645b699981cc131d24fc9b16/KqQh-dC5iG8oVzLqV0vXz.png",
        width=48,
    )
    st.title("⚙️ Configuration")
    st.divider()

    model_choice = st.radio(
        "Model to use",
        options=["🧠 SFT-Tuned (Local)", "📦 Base Model (HF)"],
        index=0,
        help="Toggle between the raw HF base model and your locally fine-tuned model.",
    )

    max_tokens = st.slider(
        "Max generation tokens",
        min_value=64,
        max_value=2048,
        value=512,
        step=64,
    )

    side_by_side = st.toggle(
        "Side-by-side comparison",
        value=False,
        help="Generate reviews from BOTH models and display them side by side.",
    )

    st.divider()
    st.caption(
        "Built with [mlx-lm](https://github.com/ml-explore/mlx-examples) "
        "and [Streamlit](https://streamlit.io)"
    )

# ── Main area ───────────────────────────────────────────────
st.title("🔬 Senior ML PR Reviewer")
st.markdown(
    "Paste a Python code snippet below and get a senior-level ML code review. "
    "Toggle models in the sidebar to compare **before** (generic LGTM) vs. "
    "**after** (sharp, specific critique)."
)

code_input = st.text_area(
    "Paste your Python code / PR diff here:",
    value=EXAMPLE_SNIPPET,
    height=280,
    key="code_input",
)

generate_btn = st.button("🚀 Generate Review", type="primary", use_container_width=True)

if generate_btn and code_input.strip():
    prompt = build_prompt(code_input)

    if side_by_side:
        col_base, col_tuned = st.columns(2)

        with col_base:
            st.subheader("📦 Base Model (Ollama)")
            with st.spinner("Generating (base)…"):
                base_review = run_ollama_inference(prompt)
            st.markdown(base_review)

        with col_tuned:
            st.subheader("🧠 SFT-Tuned Model (MLX)")
            with st.spinner("Generating (tuned)…"):
                tuned_model, tuned_tok = load_fused_model()
                tuned_review = run_mlx_inference(tuned_model, tuned_tok, prompt, max_tokens)
            st.markdown(tuned_review)
    else:
        use_tuned = "SFT" in model_choice
        label = "🧠 SFT-Tuned (MLX)" if use_tuned else "📦 Base (Ollama)"

        st.subheader(f"{label} Review")
        with st.spinner(f"Generating review with {label} model…"):
            if use_tuned:
                model, tokenizer = load_fused_model()
                review = run_mlx_inference(model, tokenizer, prompt, max_tokens)
            else:
                review = run_ollama_inference(prompt)
        st.markdown(review)

elif generate_btn:
    st.warning("Please paste some code before generating a review.")
