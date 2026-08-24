# 🔬 Senior ML PR Reviewer — Local Fine-Tuning (SFT & DPO) on Apple Silicon

> **Goal:** Fine-tune open-weight coding models (**Qwen2.5-Coder-32B** & **7B**) on Apple Silicon (M-series / MLX) to act as an automated Senior Machine Learning PR Reviewer capable of diagnosing subtle ML/DL anti-patterns with zero-shot reasoning.

---

## 📁 Project Structure

```
.
├── requirements.txt              # Minimal dependencies (mlx-lm, streamlit, ollama)
├── README.md                     # Comprehensive documentation
│
├── 🧠 SFT Pipeline (32B Model — Recommended)
│   ├── generate_diverse_dataset.py # Synthesizes 1,000+ examples across 21 ML categories
│   ├── sft_config.yaml            # mlx-lm SFT LoRA configuration for 32B model
│   └── sft_data/                  # train.jsonl (1000 ex) and valid.jsonl (100 ex)
│
├── ⚖️ DPO Pipeline (7B Experiment)
│   ├── generate_dataset.py        # Generates paired (prompt, chosen, rejected) tuples
│   ├── dpo_config.yaml            # mlx-lm DPO configuration for 7B model
│   └── data/                      # train.jsonl and valid.jsonl (DPO splits)
│
└── 🚀 Inference & UI
    └── app.py                     # Streamlit side-by-side comparison UI (Base vs Tuned)
```

---

## 🚀 Quick Start

### 1. Set Up Virtual Environment

```bash
# Create and activate a clean environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. SFT Pipeline (Qwen2.5-Coder-32B) — *Primary*

#### Step A: Generate the Diverse 21-Category Dataset
```bash
python generate_diverse_dataset.py
```
*Generates **1,000+ parameterized permutations** in `sft_data/train.jsonl` and `sft_data/valid.jsonl` covering 20 bug categories plus 100 clean negative examples.*

#### Step B: Run SFT Fine-Tuning with MLX
```bash
mlx_lm.lora --config sft_config.yaml
```
*Fits `Qwen2.5-Coder-32B-4bit` in ~48GB unified memory using LoRA (rank 8, batch size 1, grad accumulation 4).*

---

### 3. DPO Pipeline (Qwen2.5-Coder-7B) — *Experimental*

#### Step A: Generate DPO Preference Triples
```bash
python generate_dataset.py
```
*Creates `data/train.jsonl` and `data/valid.jsonl` with `(prompt, chosen, rejected)` format.*

#### Step B: Run DPO Training
```bash
mlx_lm.lora --config dpo_config.yaml --train-mode dpo
```

---

### 4. Launch the Streamlit Side-by-Side Reviewer UI

Ensure your local Ollama instance is running (`ollama run qwen2.5-coder:32b`), then launch:

```bash
streamlit run app.py
```

* **Interactive Code Input:** Paste any PyTorch or Scikit-learn PR diff.
* **Side-by-Side Comparison:** Compare the raw base model (via Ollama) against your locally tuned MLX model in real time.

---

## 🏗️ Comprehensive Anti-Pattern Taxonomy (21 Categories)

The dataset generator programmatically synthesizes diverse permutations across **20 critical ML defects** plus **negative control examples**:

### 📊 1. Data Pipeline & Leakage Bugs
| # | Anti-Pattern | Bad Pattern Summary | Correct Canonical Fix |
|---|---|---|---|
| **01** | **Data Leakage (`fit_transform`)** | Calling `scaler.fit_transform(X)` before `train_test_split` | Fit scaler on `X_train` only; transform `X_test` separately. |
| **08** | **Sequential DataLoader** | Missing `shuffle=True` on training `DataLoader` | Add `shuffle=True` to break intra-epoch batch correlation. |
| **09** | **SMOTE Leakage** | Applying `SMOTE` on dataset before cross-validation | Use `imblearn.pipeline.Pipeline` to apply SMOTE inside CV folds. |
| **10** | **Time-Series Lookahead** | Using standard random split on temporal time-series data | Use `TimeSeriesSplit` to prevent future leakage. |
| **18** | **Target Leakage in Features** | Computing target statistics on entire dataset before split | Compute target encodings using out-of-fold cross-validation. |
| **19** | **Unscaled Distance Features** | Passing unnormalized features to KNN/SVM/KMeans | Standardize features using `StandardScaler` / `RobustScaler`. |
| **20** | **Vocabulary Leakage** | Fitting `TfidfVectorizer` / Tokenizer on full corpus | Fit vectorizer on training split only; transform test split. |

### 🔥 2. PyTorch Autograd & Memory Management
| # | Anti-Pattern | Bad Pattern Summary | Correct Canonical Fix |
|---|---|---|---|
| **02** | **Graph Retention OOM** | `losses.append(loss)` retaining autograd graph | Use `losses.append(loss.item())` to free intermediate tensors. |
| **03** | **Missing `model.eval()`** | Validation loop running with active Dropout/BatchNorm | Call `model.eval()` before evaluation; restore `model.train()`. |
| **04** | **Missing `torch.no_grad()`** | Validation running without disabling gradient engine | Wrap validation block in `with torch.no_grad():`. |
| **15** | **Missing `zero_grad()`** | Accumulating gradients across steps without clearing | Call `optimizer.zero_grad()` before `loss.backward()`. |
| **16** | **In-Place Autograd Break** | Modifying tensors in-place (`x += 1`, `x.add_()`) needed for backward | Use out-of-place operations (`x = x + 1`). |
| **17** | **RNN Hidden State Retention** | Passing recurrent hidden states across batches without detaching | Call `h.detach()` between sequence chunks for Truncated BPTT. |

### 📐 3. Loss Functions & Tensor Semantics
| # | Anti-Pattern | Bad Pattern Summary | Correct Canonical Fix |
|---|---|---|---|
| **05** | **Double Sigmoid** | Adding `nn.Sigmoid()` in model and using `BCEWithLogitsLoss` | Remove model sigmoid or switch to `nn.BCELoss`. |
| **06** | **Double Softmax** | Adding `nn.Softmax()` in model and using `nn.CrossEntropyLoss` | Remove softmax layer; `CrossEntropyLoss` applies `LogSoftmax`. |
| **07** | **Silent Broadcasting Trap** | `MSELoss(preds[B, 1], targets[B])` broadcasting to `[B, B]` | Squeeze predictions: `preds.squeeze(-1)`. |
| **12** | **Augmentation on Validation** | Applying random crops/flips to validation/test datasets | Use deterministic resize/normalization for validation splits. |
| **13** | **Classification Target Dtype** | Passing `targets.float()` to `nn.CrossEntropyLoss` | Cast integer class labels to `torch.long`. |
| **14** | **Non-Contiguous `.view()`** | Calling `.view()` immediately after `.permute()` / `.transpose()` | Use `.reshape()` or `.contiguous().view()`. |

### 🎯 4. Evaluation & Bias Mitigation
| # | Anti-Pattern | Bad Pattern Summary | Correct Canonical Fix |
|---|---|---|---|
| **11** | **Accuracy on Imbalanced Data** | Evaluating 99:1 imbalanced dataset using raw accuracy | Use balanced metrics: F1, Precision-Recall AUC, or MCC. |
| **21** | **Clean Code Baseline** | Flagging false positives on correct code (model hallucination) | **Negative Examples:** Model outputs *"No critical ML bugs detected."* |

---

## 🧪 Training Configurations

### SFT 32B Config (`sft_config.yaml`)
```yaml
model: "mlx-community/Qwen2.5-Coder-32B-4bit"
train: true
data: "sft_data"
batch_size: 1
iters: 300
learning_rate: 2.0e-5
gradient_accumulation_steps: 4
adapter_path: "sft_32b_adapters"
max_seq_length: 1024
grad_checkpoint: true
num_layers: 8
```

---

## 💡 Key Engineering Lessons

1. **SFT Before DPO:** DPO cannot establish output schemas from scratch. An SFT warm-start is mandatory to establish domain syntax and tone.
2. **Base Models $\neq$ Chat Models:** Base models hallucinate system prompts when wrapped in `<|im_start|>` chat templates. Strip templates and feed raw completion formats.
3. **Negative Sampling is Mandatory:** Models trained solely on buggy code develop an extreme false-positive bias. Injecting 10% clean negative examples eliminates false alarms and forces semantic reasoning.

---

## 📄 License
MIT License.
