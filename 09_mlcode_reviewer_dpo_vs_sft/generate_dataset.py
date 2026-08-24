#!/usr/bin/env python3
"""
generate_dataset.py – Synthetic DPO dataset for Senior ML PR Reviewer

Generates paired (chosen, rejected) examples for three ML anti-patterns:
  1. Data Leakage:      fit_transform() before train_test_split
  2. PyTorch OOM Leak:  accumulating `loss` instead of `loss.item()`
  3. Eval Mode Forgot:  missing model.eval() / torch.no_grad() in validation

Output: data/train.jsonl and data/valid.jsonl  (80/20 split)
Each row: {"prompt": ..., "chosen": ..., "rejected": ...}
"""

import json
import os
import random
from pathlib import Path

# ── reproducibility ─────────────────────────────────────────
random.seed(42)

# ── output paths ────────────────────────────────────────────
DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "sft_train.jsonl"
VALID_PATH = DATA_DIR / "sft_valid.jsonl"


# ════════════════════════════════════════════════════════════
#  Template bank – each anti-pattern has multiple variants
#  so the model sees linguistic diversity, not memorisation.
# ════════════════════════════════════════════════════════════

# ── Generic "rejected" reviews (the lazy reviewer) ─────────
GENERIC_REVIEWS = [
    # ── Category 1: The Lazy Reviewer (Too Short) ──
    "LGTM! Nice variable names, maybe add some docstrings.",
    "Looks good to me! Clean code. 👍",
    "No issues from me, ship it!",
    "Approved. Code is clear and well-structured.",
    
    # ── Category 2: The Yapping Hallucinator (Too Long / Fake Issues) ──
    """Feedback on ML-Specific Correctness:

Optimizer Initialization:
Issue: The optimizer is initialized with the model's parameters, but the learning rate is fixed. This might not be optimal.
Recommendation: Consider adding a learning rate scheduler to adjust the learning rate during training.

Batch Normalization and Dropout:
Issue: The model does not include any batch normalization or dropout layers.
Recommendation: You should add BatchNorm and Dropout to prevent overfitting, even if the model is simple.

Return Statements:
Issue: The training step doesn't explicitly return the predictions, making it hard to monitor.
Recommendation: Always return predictions along with the loss for debugging purposes.

Overall, make sure you address these structural issues before merging!""",

    """Review for ML correctness:

1. Data Types: Ensure that your inputs are cast to `torch.float32`. Sometimes tensors are implicitly `float16` or `float64`, which can cause device mismatch errors or numerical instability during backpropagation.
2. Device Consistency: You move `batch_x` and `batch_y` to `.cuda()`, but you did not explicitly move the `predictions` output to `.cuda()`. You should write `predictions = predictions.cuda()` to prevent device mismatches when computing the loss.
3. Magic Numbers: You are using `128` and `512` in your linear layers. Consider moving these to a config file.
4. Validation: There is no validation loop here. You should always have a validation loop.

By fixing these points, the model will be much more robust.""",

    """Here are a few suggestions for your ML code:

- Loss Function: While you are using a standard loss function, consider if a custom loss function might better suit your specific domain logic.
- Dataloader: Make sure your DataLoader has `pin_memory=True`. While you might have it, it's not explicitly visible in this diff.
- Type Hints: There are no type hints on your tensors. Using `torch.Tensor` type hints makes the code much cleaner.
- Activation Functions: ReLU is standard, but you might want to try GELU or Swish as they sometimes perform slightly better on deep networks.

Looks solid otherwise, great job!"""
]


# ───────────────────────────────────────────────────────────
#  ANTI-PATTERN 1 — Data Leakage  (fit_transform before split)
# ───────────────────────────────────────────────────────────

DATA_LEAKAGE_DIFFS = [
    # Variant A – StandardScaler
    {
        "diff": """\
```diff
+ from sklearn.preprocessing import StandardScaler
+ from sklearn.model_selection import train_test_split
+
+ scaler = StandardScaler()
+ X_scaled = scaler.fit_transform(X)      # <── scales on ALL data
+ X_train, X_test, y_train, y_test = train_test_split(
+     X_scaled, y, test_size=0.2, random_state=42
+ )
+ model.fit(X_train, y_train)
+ print("Test accuracy:", model.score(X_test, y_test))
```""",
        "feature": "feature/preprocessing-pipeline",
        "scaler": "StandardScaler",
    },
    # Variant B – MinMaxScaler
    {
        "diff": """\
```diff
+ from sklearn.preprocessing import MinMaxScaler
+ from sklearn.model_selection import train_test_split
+
+ mm = MinMaxScaler()
+ df[features] = mm.fit_transform(df[features])  # <── leaks test stats
+ train_df, test_df = train_test_split(df, test_size=0.25)
+ clf.fit(train_df[features], train_df["label"])
+ print(clf.score(test_df[features], test_df["label"]))
```""",
        "feature": "feature/normalize-inputs",
        "scaler": "MinMaxScaler",
    },
    # Variant C – RobustScaler + Pipeline context
    {
        "diff": """\
```diff
+ from sklearn.preprocessing import RobustScaler
+ from sklearn.model_selection import train_test_split
+ import numpy as np
+
+ rs = RobustScaler()
+ X = rs.fit_transform(raw_features)
+ X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.15)
+ regressor.fit(X_train, y_train)
+ val_rmse = np.sqrt(np.mean((regressor.predict(X_val) - y_val) ** 2))
+ print(f"Validation RMSE: {val_rmse:.4f}")
```""",
        "feature": "feature/robust-scaling",
        "scaler": "RobustScaler",
    },
    # Variant D – QuantileTransformer
    {
        "diff": """\
```diff
+ from sklearn.preprocessing import QuantileTransformer
+ from sklearn.model_selection import train_test_split
+
+ qt = QuantileTransformer(output_distribution="normal")
+ X_transformed = qt.fit_transform(X_raw)
+ X_tr, X_te, y_tr, y_te = train_test_split(X_transformed, y, test_size=0.2)
+ model.fit(X_tr, y_tr)
+ print("AUC:", roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
```""",
        "feature": "feature/quantile-normalize",
        "scaler": "QuantileTransformer",
    },
    # Variant E – PowerTransformer
    {
        "diff": """\
```diff
+ from sklearn.preprocessing import PowerTransformer
+ from sklearn.model_selection import train_test_split
+
+ pt = PowerTransformer(method="yeo-johnson")
+ X_norm = pt.fit_transform(X)
+ X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2)
+ forest = RandomForestClassifier(n_estimators=200)
+ forest.fit(X_train, y_train)
+ print("F1:", f1_score(y_test, forest.predict(X_test), average="macro"))
```""",
        "feature": "feature/power-transform",
        "scaler": "PowerTransformer",
    },
    # Variant F – MaxAbsScaler in sparse context
    {
        "diff": """\
```diff
+ from sklearn.preprocessing import MaxAbsScaler
+ from sklearn.model_selection import train_test_split
+
+ scaler = MaxAbsScaler()
+ X_scaled = scaler.fit_transform(X_sparse)
+ X_train, X_test, y_train, y_test = train_test_split(
+     X_scaled, y, test_size=0.3, stratify=y
+ )
+ svm = LinearSVC()
+ svm.fit(X_train, y_train)
```""",
        "feature": "feature/sparse-scaling",
        "scaler": "MaxAbsScaler",
    },
    # Variant G – Normalizer (L2)
    {
        "diff": """\
```diff
+ from sklearn.preprocessing import Normalizer
+ from sklearn.model_selection import train_test_split
+
+ norm = Normalizer(norm="l2")
+ X_normed = norm.fit_transform(embeddings)
+ train_X, test_X, train_y, test_y = train_test_split(X_normed, labels)
+ knn = KNeighborsClassifier(n_neighbors=5)
+ knn.fit(train_X, train_y)
+ print("KNN accuracy:", knn.score(test_X, test_y))
```""",
        "feature": "feature/embedding-normalization",
        "scaler": "Normalizer",
    },
    # Variant H – TfidfVectorizer (NLP leakage)
    {
        "diff": """\
```diff
+ from sklearn.feature_extraction.text import TfidfVectorizer
+ from sklearn.model_selection import train_test_split
+
+ tfidf = TfidfVectorizer(max_features=10000)
+ X = tfidf.fit_transform(corpus)
+ X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)
+ clf = LogisticRegression(max_iter=1000)
+ clf.fit(X_train, y_train)
+ print("Accuracy:", clf.score(X_test, y_test))
```""",
        "feature": "feature/tfidf-classifier",
        "scaler": "TfidfVectorizer",
    },
    # Variant I – LabelEncoder on target (subtle)
    {
        "diff": """\
```diff
+ from sklearn.preprocessing import LabelEncoder, StandardScaler
+ from sklearn.model_selection import train_test_split
+
+ le = LabelEncoder()
+ scaler = StandardScaler()
+ y_enc = le.fit_transform(y)
+ X_scaled = scaler.fit_transform(X)
+ X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2)
+ model.fit(X_train, y_train)
```""",
        "feature": "feature/encode-and-scale",
        "scaler": "StandardScaler",
    },
]

DATA_LEAKAGE_CHOSEN_TEMPLATES = [
    """\
🚨 **Critical: Data Leakage – {scaler} fit on full dataset before split**

You call `{scaler}.fit_transform()` on the **entire dataset** before `train_test_split`. \
This means the test set's distribution statistics (mean, variance, quantiles) are baked \
into the training data. Your reported metric is **artificially inflated** and will not \
generalise to production data.

**Why this is dangerous:**
- The scaler learns global statistics that include test-set information.
- Metric improvements in CI/CD will be illusory; real-world performance will be worse.
- In time-series or medical settings this can be a regulatory violation.

**Fix – fit only on training data, transform test separately:**
```diff
- X_scaled = scaler.fit_transform(X)
- X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, ...)
+ X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
+ scaler.fit(X_train)
+ X_train = scaler.transform(X_train)
+ X_test  = scaler.transform(X_test)
```

Or better yet, use `sklearn.pipeline.Pipeline` so the transform is bound to the \
cross-validation fold automatically.""",

    """\
🛑 **Data Leakage** detected in `{feature}`.

`fit_transform()` is called **before** the train/test split. The scaler's learned \
parameters (e.g. mean, scale) are contaminated with test-set statistics, making your \
evaluation metric unreliable.

**Impact:** Expect real-world performance to be *significantly* worse than reported. \
This is the #1 most common silent bug in ML pipelines.

**Suggested fix:**
```diff
- scaled = scaler.fit_transform(full_data)
- X_train, X_test = train_test_split(scaled, ...)
+ X_train, X_test = train_test_split(full_data, ...)
+ scaler.fit(X_train)
+ X_train = scaler.transform(X_train)
+ X_test  = scaler.transform(X_test)
```

Consider wrapping with `sklearn.pipeline.Pipeline` + `cross_val_score` for a \
leak-proof setup.""",
]


# ───────────────────────────────────────────────────────────
#  ANTI-PATTERN 2 — PyTorch OOM Leak  (loss vs loss.item())
# ───────────────────────────────────────────────────────────

OOM_LEAK_DIFFS = [
    # Variant A – list append
    {
        "diff": """\
```diff
+ losses = []
+ for epoch in range(num_epochs):
+     for batch in dataloader:
+         optimizer.zero_grad()
+         output = model(batch["input_ids"])
+         loss = criterion(output, batch["labels"])
+         loss.backward()
+         optimizer.step()
+         losses.append(loss)        # <── graph kept alive
+     print(f"Epoch {epoch} avg loss: {sum(losses)/len(losses)}")
```""",
        "var": "losses",
        "feature": "feature/training-loop",
    },
    # Variant B – running sum
    {
        "diff": """\
```diff
+ total_loss = 0
+ for step, batch in enumerate(train_loader):
+     logits = model(batch.x)
+     loss = F.cross_entropy(logits, batch.y)
+     loss.backward()
+     optimizer.step()
+     optimizer.zero_grad()
+     total_loss += loss             # <── accumulates graph nodes
+ avg = total_loss / (step + 1)
+ print(f"Train loss: {avg:.4f}")
```""",
        "var": "total_loss",
        "feature": "feature/gnn-train",
    },
    # Variant C – logging dict
    {
        "diff": """\
```diff
+ metrics = {"loss": [], "lr": []}
+ for i, (x, y) in enumerate(loader):
+     pred = net(x.to(device))
+     loss = loss_fn(pred, y.to(device))
+     loss.backward()
+     opt.step()
+     opt.zero_grad()
+     metrics["loss"].append(loss)   # <── tensor, not scalar
+     metrics["lr"].append(scheduler.get_last_lr()[0])
```""",
        "var": 'metrics["loss"]',
        "feature": "feature/metric-logging",
    },
    # Variant D – wandb log
    {
        "diff": """\
```diff
+ epoch_losses = []
+ for data, target in trainloader:
+     optimizer.zero_grad()
+     output = model(data.cuda())
+     loss = criterion(output, target.cuda())
+     loss.backward()
+     optimizer.step()
+     epoch_losses.append(loss)      # <── full tensor stored
+ wandb.log({"train_loss": sum(epoch_losses) / len(epoch_losses)})
```""",
        "var": "epoch_losses",
        "feature": "feature/wandb-integration",
    },
    # Variant E – tensorboard logging
    {
        "diff": """\
```diff
+ from torch.utils.tensorboard import SummaryWriter
+ writer = SummaryWriter()
+ all_losses = []
+ for step, (inputs, targets) in enumerate(train_dl):
+     out = model(inputs)
+     loss = criterion(out, targets)
+     loss.backward()
+     optimizer.step()
+     optimizer.zero_grad()
+     all_losses.append(loss)         # <── keeps graph
+     if step % 100 == 0:
+         writer.add_scalar("loss", sum(all_losses[-100:]) / 100, step)
```""",
        "var": "all_losses",
        "feature": "feature/tensorboard-logging",
    },
    # Variant F – tqdm progress bar
    {
        "diff": """\
```diff
+ from tqdm import tqdm
+ running_loss = 0.0
+ pbar = tqdm(train_loader)
+ for batch in pbar:
+     optimizer.zero_grad()
+     preds = model(batch["x"].to(device))
+     loss = F.mse_loss(preds, batch["y"].to(device))
+     loss.backward()
+     optimizer.step()
+     running_loss += loss             # <── graph node accumulation
+     pbar.set_postfix(loss=running_loss.item() / (pbar.n + 1))
```""",
        "var": "running_loss",
        "feature": "feature/tqdm-progress",
    },
    # Variant G – distributed training
    {
        "diff": """\
```diff
+ step_losses = []
+ for batch_idx, batch in enumerate(distributed_loader):
+     with autocast():
+         output = ddp_model(batch["input"].cuda())
+         loss = loss_fn(output, batch["target"].cuda())
+     scaler.scale(loss).backward()
+     scaler.step(optimizer)
+     scaler.update()
+     optimizer.zero_grad()
+     step_losses.append(loss)        # <── retains scaled graph
+     if dist.get_rank() == 0:
+         print(f"Step {batch_idx}: {loss:.4f}")
```""",
        "var": "step_losses",
        "feature": "feature/ddp-training",
    },
    # Variant H – GAN discriminator
    {
        "diff": """\
```diff
+ d_losses = []
+ for real_imgs, _ in dataloader:
+     real_imgs = real_imgs.to(device)
+     fake_imgs = generator(torch.randn(batch_size, latent_dim, device=device))
+     d_real = discriminator(real_imgs)
+     d_fake = discriminator(fake_imgs.detach())
+     d_loss = F.binary_cross_entropy(d_real, ones) + F.binary_cross_entropy(d_fake, zeros)
+     d_loss.backward()
+     d_optimizer.step()
+     d_optimizer.zero_grad()
+     d_losses.append(d_loss)         # <── graph retained
```""",
        "var": "d_losses",
        "feature": "feature/gan-training",
    },
    # Variant I – validation loss accumulation (subtle)
    {
        "diff": """\
```diff
+ val_loss_sum = 0
+ model.eval()
+ with torch.no_grad():
+     for x, y in val_loader:
+         out = model(x.to(device))
+         loss = criterion(out, y.to(device))
+         val_loss_sum += loss           # <── tensor accumulation
+ print(f"Val loss: {val_loss_sum / len(val_loader):.4f}")
+ model.train()
```""",
        "var": "val_loss_sum",
        "feature": "feature/val-loop",
    },
]

OOM_LEAK_CHOSEN_TEMPLATES = [
    """\
🔥 **Memory Leak: Computation graph retained via `{var}`**

You are appending `loss` (a full `torch.Tensor` with `grad_fn`) to `{var}`. \
Every tensor you keep **pins the entire computation graph in GPU/MPS memory**. \
After a few hundred steps, you will OOM.

**Why this is dangerous:**
- Each `loss` tensor holds a reference chain back through every intermediate activation.
- Memory grows **linearly per step** instead of staying constant.
- The OOM crash is non-deterministic and very hard to debug in production.

**Fix – detach to a Python float:**
```diff
- {var}.append(loss)
+ {var}.append(loss.item())
```

`loss.item()` extracts the scalar value and lets PyTorch free the graph.""",

    """\
⚠️ **PyTorch OOM bug** in `{feature}`.

Storing `loss` directly into `{var}` keeps the **entire autograd graph alive**. \
This is a textbook memory leak — memory usage will climb linearly with training \
steps until the process is killed.

**The fix is one character:**
```diff
- {var}.append(loss)
+ {var}.append(loss.item())
```

`.item()` converts the 0-d tensor to a Python `float`, releasing the graph. \
Also consider logging every N steps instead of every step to reduce overhead.""",
]


# ───────────────────────────────────────────────────────────
#  ANTI-PATTERN 3 — Missing model.eval() / torch.no_grad()
# ───────────────────────────────────────────────────────────

EVAL_MODE_DIFFS = [
    # Variant A – basic classification
    {
        "diff": """\
```diff
+ # Validation
+ val_loss = 0
+ correct = 0
+ for batch in val_loader:
+     output = model(batch["input_ids"].to(device))
+     val_loss += F.cross_entropy(output, batch["labels"].to(device)).item()
+     correct += (output.argmax(1) == batch["labels"].to(device)).sum().item()
+ print(f"Val accuracy: {correct / len(val_dataset):.2%}")
```""",
        "model_name": "model",
        "feature": "feature/add-validation",
    },
    # Variant B – regression with dropout
    {
        "diff": """\
```diff
+ def evaluate(model, loader, device):
+     preds, targets = [], []
+     for x, y in loader:
+         out = model(x.to(device))
+         preds.append(out.detach().cpu())
+         targets.append(y)
+     preds = torch.cat(preds)
+     targets = torch.cat(targets)
+     return torch.sqrt(F.mse_loss(preds, targets))
```""",
        "model_name": "model",
        "feature": "feature/eval-rmse",
    },
    # Variant C – NLP inference
    {
        "diff": """\
```diff
+ results = []
+ for text in test_texts:
+     tokens = tokenizer(text, return_tensors="pt").to(device)
+     logits = model(**tokens).logits
+     label = logits.argmax(dim=-1).item()
+     results.append({"text": text, "prediction": label})
```""",
        "model_name": "model",
        "feature": "feature/batch-inference",
    },
    # Variant D – generative model
    {
        "diff": """\
```diff
+ val_losses = []
+ for batch in val_dataloader:
+     input_ids = batch["input_ids"].to(device)
+     labels = batch["labels"].to(device)
+     outputs = model(input_ids=input_ids, labels=labels)
+     val_losses.append(outputs.loss.item())
+ avg_val_loss = sum(val_losses) / len(val_losses)
+ print(f"Validation loss: {avg_val_loss:.4f}")
```""",
        "model_name": "model",
        "feature": "feature/causal-lm-eval",
    },
    # Variant E – image classifier
    {
        "diff": """\
```diff
+ top1_correct = 0
+ total = 0
+ for images, labels in test_loader:
+     images, labels = images.to(device), labels.to(device)
+     outputs = model(images)
+     _, predicted = outputs.max(1)
+     total += labels.size(0)
+     top1_correct += predicted.eq(labels).sum().item()
+ print(f"Top-1 accuracy: {100. * top1_correct / total:.2f}%")
```""",
        "model_name": "model",
        "feature": "feature/test-accuracy",
    },
    # Variant F – speech recognition
    {
        "diff": """\
```diff
+ wer_scores = []
+ for audio, transcript in eval_dataset:
+     mel = feature_extractor(audio).to(device)
+     logits = speech_model(mel)
+     decoded = decoder(logits)
+     wer_scores.append(compute_wer(decoded, transcript))
+ print(f"Average WER: {sum(wer_scores)/len(wer_scores):.4f}")
```""",
        "model_name": "speech_model",
        "feature": "feature/speech-eval",
    },
    # Variant G – object detection mAP
    {
        "diff": """\
```diff
+ all_detections = []
+ for images, targets in val_loader:
+     images = [img.to(device) for img in images]
+     predictions = detector(images)
+     all_detections.extend(predictions)
+ mAP = compute_mAP(all_detections, ground_truth)
+ print(f"Validation mAP@0.5: {mAP:.4f}")
```""",
        "model_name": "detector",
        "feature": "feature/detection-eval",
    },
    # Variant H – reinforcement learning policy
    {
        "diff": """\
```diff
+ rewards = []
+ for episode in range(100):
+     obs = env.reset()
+     done = False
+     total_reward = 0
+     while not done:
+         action = policy_net(torch.tensor(obs, device=device).float())
+         obs, reward, done, _ = env.step(action.argmax().item())
+         total_reward += reward
+     rewards.append(total_reward)
+ print(f"Mean reward: {sum(rewards)/len(rewards):.2f}")
```""",
        "model_name": "policy_net",
        "feature": "feature/rl-evaluation",
    },
    # Variant I – multi-task model
    {
        "diff": """\
```diff
+ cls_preds, reg_preds = [], []
+ for batch in eval_loader:
+     x = batch["features"].to(device)
+     cls_out, reg_out = multi_model(x)
+     cls_preds.append(cls_out.argmax(1).cpu())
+     reg_preds.append(reg_out.cpu())
+ cls_acc = (torch.cat(cls_preds) == eval_labels).float().mean()
+ reg_mae = (torch.cat(reg_preds) - eval_targets).abs().mean()
+ print(f"CLS acc: {cls_acc:.4f}, REG MAE: {reg_mae:.4f}")
```""",
        "model_name": "multi_model",
        "feature": "feature/multi-task-eval",
    },
]

EVAL_MODE_CHOSEN_TEMPLATES = [
    """\
🐛 **Bug: Validation runs in training mode — dropout & batchnorm are active**

There is no `{model_name}.eval()` or `torch.no_grad()` guard before the validation \
loop. This means:
- **Dropout** randomly zeros activations → validation metrics are noisy & non-reproducible.
- **BatchNorm** uses per-batch statistics instead of the learned running mean/var.
- **Gradients are still tracked**, wasting ~30-40% memory and slowing inference.

**Fix:**
```diff
+ {model_name}.eval()
+ with torch.no_grad():
      for batch in val_loader:
          ...
+ {model_name}.train()   # restore for next training epoch
```

This is especially critical if you are using validation loss for **early stopping** \
or **learning-rate scheduling** — stochastic dropout will inject noise into those \
decisions.""",

    """\
⚠️ **Missing `eval()` / `no_grad()` in validation** (`{feature}`)

The model is evaluated while still in `.train()` mode. Consequences:
1. **Dropout** is active → metrics fluctuate between runs with the same data.
2. **BatchNorm** uses mini-batch stats, not the learned population stats.
3. **Autograd graph** is unnecessarily built → 30-40 % extra memory + slower.

**Suggested fix:**
```diff
+ {model_name}.eval()
+ with torch.no_grad():
      for batch in val_loader:
          ...
+ {model_name}.train()
```

If you are saving checkpoints based on "best val loss", the noisy metric can cause \
you to save a sub-optimal checkpoint.""",
]


# ════════════════════════════════════════════════════════════
#  Assembly – build prompt/chosen/rejected triples
# ════════════════════════════════════════════════════════════

PROMPT_TEMPLATE = """\
You are a senior Machine Learning engineer reviewing a GitHub Pull Request.
Review the following code diff from branch `{feature}` and provide detailed, \
actionable feedback. Focus on ML-specific correctness, not style.

{diff}"""


# ───────────────────────────────────────────────────────────
#  ANTI-PATTERN 4 — SMOTE Cross-Validation Leakage
# ───────────────────────────────────────────────────────────

SMOTE_LEAK_DIFFS = [
    {
        "diff": """\
```diff
+ from imblearn.over_sampling import SMOTE
+ from sklearn.model_selection import cross_val_score
+ from sklearn.ensemble import RandomForestClassifier
+
+ smote = SMOTE(random_state=42)
+ X_res, y_res = smote.fit_resample(X, y)
+ 
+ model = RandomForestClassifier()
+ scores = cross_val_score(model, X_res, y_res, cv=5, scoring='f1')
+ print("F1:", scores.mean())
```""",
        "feature": "feature/smote-leakage",
        "oversampler": "SMOTE",
    },
    {
        "diff": """\
```diff
+ from imblearn.over_sampling import ADASYN
+ from sklearn.model_selection import KFold
+ from xgboost import XGBClassifier
+
+ adasyn = ADASYN()
+ X_synth, y_synth = adasyn.fit_resample(X_train, y_train)
+ 
+ kf = KFold(n_splits=3)
+ for train_idx, val_idx in kf.split(X_synth):
+     X_f_train, X_f_val = X_synth[train_idx], X_synth[val_idx]
+     y_f_train, y_f_val = y_synth[train_idx], y_synth[val_idx]
+     model = XGBClassifier().fit(X_f_train, y_f_train)
```""",
        "feature": "feature/adasyn-cv",
        "oversampler": "ADASYN",
    },
]

SMOTE_LEAK_CHOSEN_TEMPLATES = [
    "**1. Cross-Validation Data Leakage**: You are applying `{oversampler}` on the entire dataset *before* performing cross-validation (or splitting the folds). This means synthetic samples generated using information from the validation folds leak into the training folds, artificially inflating your F1 scores. \n\n**Fix**: You must apply oversampling *inside* the cross-validation loop. Use an `imblearn.pipeline.Pipeline` instead so that SMOTE is only applied to the training folds at each split.",
    "**Critical ML Anti-Pattern: SMOTE Leakage**\nBy applying oversampling before your CV splits, you are leaking data. Synthetic data points in the training set are heavily influenced by points that end up in the validation set, which completely breaks the independence of your validation folds.\n\n**Action**: Wrap your model and `{oversampler}` inside an `imblearn.pipeline.Pipeline` and pass that pipeline to `cross_val_score`.",
]


# ───────────────────────────────────────────────────────────
#  ANTI-PATTERN 5 — Time-Series Data Leakage (Improper Splitting)
# ───────────────────────────────────────────────────────────

TIME_SERIES_LEAK_DIFFS = [
    {
        "diff": """\
```diff
+ import pandas as pd
+ from sklearn.model_selection import train_test_split
+
+ df = pd.read_csv("stock_prices.csv").sort_values("Date")
+ X = df[["Open", "Volume"]]
+ y = df["Close"]
+
+ X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
+ model.fit(X_train, y_train)
```""",
        "feature": "feature/stock-model",
    },
    {
        "diff": """\
```diff
+ df = load_timeseries_data()
+ # df is sorted chronologically
+ features = df.drop(columns=['target'])
+ target = df['target']
+ 
+ train_x, test_x, train_y, test_y = train_test_split(
+     features, target, random_state=123, shuffle=True
+ )
+ xgb.fit(train_x, train_y)
```""",
        "feature": "feature/timeseries-forecasting",
    },
]

TIME_SERIES_LEAK_CHOSEN_TEMPLATES = [
    "**1. Time-Series Data Leakage**: You are using `train_test_split` with shuffling (which is the default) on time-series data. This randomly distributes data points across the training and test sets, meaning the model is being trained on *future* data to predict *past* data.\n\n**Fix**: You must split the data sequentially. Set `shuffle=False` in `train_test_split`, or use `sklearn.model_selection.TimeSeriesSplit`.",
    "**Major Issue: Temporal Leakage**\nWhen dealing with sequential or time-series data, you cannot randomly shuffle the rows. By doing so, you're leaking future information into the training set, which will give you wildly over-optimistic performance metrics.\n\n**Fix**: Use a chronological split by either passing `shuffle=False` to `train_test_split` or indexing the dataframe manually based on a cutoff date.",
]


# ───────────────────────────────────────────────────────────
#  ANTI-PATTERN 6 — Deceptive Metrics on Imbalanced Data
# ───────────────────────────────────────────────────────────

DECEPTIVE_METRIC_DIFFS = [
    {
        "diff": """\
```diff
+ # highly imbalanced dataset (99% negative)
+ X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
+ 
+ model = LogisticRegression()
+ model.fit(X_train, y_train)
+ 
+ acc = accuracy_score(y_test, model.predict(X_test))
+ print(f"Model Accuracy: {acc * 100:.2f}%")
+ if acc > 0.95:
+     print("Model performing exceptionally well!")
```""",
        "feature": "feature/fraud-detection",
    },
    {
        "diff": """\
```diff
+ # anomaly detection (1 in 10000 anomalies)
+ clf = IsolationForest()
+ clf.fit(X_train)
+ preds = clf.predict(X_test)
+ 
+ # Convert -1 to 1, 1 to 0
+ preds = [1 if p == -1 else 0 for p in preds]
+ 
+ print("Accuracy:", (preds == y_test).mean())
```""",
        "feature": "feature/anomaly-eval",
    },
]

DECEPTIVE_METRIC_CHOSEN_TEMPLATES = [
    "**1. Deceptive Metric for Imbalanced Data**: You are evaluating the model using raw `accuracy` on a highly imbalanced dataset. Since the majority class dominates the dataset, a dummy model that predicts 0 for everything would achieve near-perfect accuracy, masking the fact that the model is completely failing to detect the minority class.\n\n**Fix**: Use metrics that are robust to class imbalance, such as F1-score, Precision-Recall AUC (PR-AUC), or Balanced Accuracy.",
    "**Evaluation Flaw**: Accuracy is fundamentally misleading for datasets where the classes are highly skewed (like anomaly or fraud detection). A 99% accuracy might just mean the model always predicts the negative class.\n\n**Fix**: Replace `accuracy` with `sklearn.metrics.f1_score`, `roc_auc_score`, or plot a Confusion Matrix to see the true positive and false positive rates.",
]


# ───────────────────────────────────────────────────────────
#  ANTI-PATTERN 7 — PyTorch / Deep Learning Fatal Traps
# ───────────────────────────────────────────────────────────

PYTORCH_FLAWS_DIFFS = [
    {
        "diff": """\
```diff
+ class Model(nn.Module):
+     def __init__(self):
+         super().__init__()
+         self.fc1 = nn.Linear(768, 128)
+         self.fc2 = nn.Linear(128, 10)
+         self.softmax = nn.Softmax(dim=1)
+         
+     def forward(self, x):
+         x = F.relu(self.fc1(x))
+         return self.softmax(self.fc2(x))
+ 
+ criterion = nn.CrossEntropyLoss()
+ output = model(inputs)
+ loss = criterion(output, labels)
```""",
        "feature": "feature/classification-model",
    },
    {
        "diff": """\
```diff
+ optimizer = optim.Adam(model.parameters(), lr=1e-3)
+ for epoch in range(10):
+     for batch_x, batch_y in dataloader:
+         outputs = model(batch_x)
+         loss = criterion(outputs, batch_y)
+         loss.backward()
+         optimizer.step()
```""",
        "feature": "feature/training-loop",
    },
    {
        "diff": """\
```diff
+ train_dataset = ImageFolder("data/train", transform=transform)
+ train_loader = DataLoader(train_dataset, batch_size=256)
+ 
+ for images, labels in train_loader:
+     images, labels = images.cuda(), labels.cuda()
+     outputs = model(images)
```""",
        "feature": "feature/vision-dataloader",
    },
    {
        "diff": """\
```diff
+ x = x.permute(0, 2, 1)  # shape [batch, features, seq_len]
+ x = x.view(batch_size, -1)
+ out = self.fc(x)
```""",
        "feature": "feature/transformer-head",
    },
]

PYTORCH_FLAWS_CHOSEN_TEMPLATES = [
    "**Major PyTorch Bug Detected**: Your code contains a critical mathematical or structural flaw in the PyTorch graph.\n\nIf this is the double Softmax bug: `nn.CrossEntropyLoss` already applies `LogSoftmax` internally. Applying `nn.Softmax()` before it will completely flatten your gradients. \nIf this is a missing `optimizer.zero_grad()`: Your gradients will accumulate forever, causing weights to explode.\nIf this is a DataLoader issue: You are missing `num_workers` and `pin_memory=True`, which will cause massive GPU starvation.\nIf this is a reshape issue: Using `.view()` on a permuted tensor will scramble the memory layout silently. Use `.reshape()` instead.",
    "🚨 **Fatal PyTorch Anti-Pattern**\nYou've made a classic deep learning mistake that will either stop the model from learning entirely or crater your performance.\n\n**Action**: Review your activation functions against your loss function, ensure your training loop zeroes gradients, use `.reshape()` for permuted tensors, and always use `num_workers>0` for image dataloaders."
]

# ───────────────────────────────────────────────────────────
#  ANTI-PATTERN 8 — Advanced Data Leakage
# ───────────────────────────────────────────────────────────

ADVANCED_LEAKAGE_DIFFS = [
    {
        "diff": """\
```diff
+ df['age'].fillna(df['age'].mean(), inplace=True)
+ X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'])
+ model.fit(X_train, y_train)
```""",
        "feature": "feature/imputation",
    },
    {
        "diff": """\
```diff
+ # target is 'will_churn' (boolean)
+ features = ['age', 'monthly_spend', 'total_days_active_before_churn']
+ X_train, X_test, y_train, y_test = train_test_split(df[features], df['will_churn'])
+ model.fit(X_train, y_train)
```""",
        "feature": "feature/churn-model",
    },
]

ADVANCED_LEAKAGE_CHOSEN_TEMPLATES = [
    "**Data Leakage Warning**: You are leaking data from the test set into the training set.\n\nIf you are imputing missing values: You applied `.fillna(mean)` on the *entire* dataset before splitting. The test set's mean is now baked into the training data.\nIf you have target leakage: You included a feature (`total_days_active_before_churn`) that perfectly proxies the target variable. The model will get 99% accuracy offline and fail instantly in production.",
    "**Critical Preprocessing Flaw**\nThis code suffers from look-ahead bias / data leakage. You must perform all imputations *after* the train/test split, fitting the imputer only on the training data. Furthermore, ensure no features inherently look ahead into the future of the target variable."
]

# ───────────────────────────────────────────────────────────
#  ANTI-PATTERN 9 — Classical ML Errors
# ───────────────────────────────────────────────────────────

CLASSICAL_ML_DIFFS = [
    {
        "diff": """\
```diff
+ from sklearn.svm import SVC
+ 
+ X_train, X_test, y_train, y_test = train_test_split(X, y)
+ 
+ model = SVC(kernel='rbf')
+ model.fit(X_train, y_train)
+ print(model.score(X_test, y_test))
```""",
        "feature": "feature/svm-classifier",
    },
    {
        "diff": """\
```diff
+ import pandas as pd
+ import xgboost as xgb
+ 
+ df = pd.get_dummies(df, columns=['zip_code']) # 40,000 unique zip codes
+ X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'])
+ 
+ model = xgb.XGBClassifier()
+ model.fit(X_train, y_train)
```""",
        "feature": "feature/xgboost-geo",
    },
]

CLASSICAL_ML_CHOSEN_TEMPLATES = [
    "**Classical ML Anti-Pattern**: You are misusing a foundational ML algorithm.\n\nIf using a distance-based model (SVM, KNN, KMeans): You forgot to scale the features. Features with larger numerical ranges will completely dominate the distance calculations.\nIf using Tree models (XGBoost, RF): You used One-Hot Encoding on a high-cardinality categorical variable (like zip code). This creates a massive, sparse matrix that destroys the tree's ability to make meaningful splits. Use Target Encoding or XGBoost's native categorical support instead.",
    "🚨 **Algorithm Misuse**\nDistance-based models require `StandardScaler` or `MinMaxScaler`. Tree-based models fail on heavily sparse one-hot encoded features. Fix your preprocessing to match the specific algorithm's mathematical assumptions."
]

# ───────────────────────────────────────────────────────────
#  ANTI-PATTERN 10 — NLP Specific Traps
# ───────────────────────────────────────────────────────────

NLP_FLAWS_DIFFS = [
    {
        "diff": """\
```diff
+ criterion = nn.CrossEntropyLoss()
+ 
+ for batch in dataloader:
+     input_ids = batch['input_ids'].cuda()
+     labels = batch['labels'].cuda()
+     
+     logits = model(input_ids)
+     loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
+     loss.backward()
```""",
        "feature": "feature/llm-finetuning",
    }
]

NLP_FLAWS_CHOSEN_TEMPLATES = [
    "**NLP Loss Flaw**: You are calculating `CrossEntropyLoss` over the entire sequence without ignoring padding tokens. Because sequences are padded to a max length, the model will waste all its capacity learning to confidently predict `[PAD]` tokens, dragging down actual performance.\n\n**Fix**: Initialize your loss function with `nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)`.",
    "**Critical NLP Bug: Padding Token Loss**\nYour loss function includes the padding tokens. This artificially drives down the loss and causes the model to optimize for padding rather than the actual sequence. Pass `ignore_index` to your criterion to mask out the padding."
]

OOD_FLAWS_DIFFS = [
    {
        "diff": """```diff
  train_dataset = datasets.ImageFolder("data/train", transform=transform)
- train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, pin_memory=True)
+ train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, pin_memory=True, shuffle=True)
```""",
        "model_name": "resnet",
        "feature": "feature/dataloader-shuffle",
    },
    {
        "diff": """```diff
      def forward(self, x):
          x = torch.relu(self.fc1(x))
          x = self.fc2(x)
-         return torch.sigmoid(x)
+         return x
  
  criterion = nn.BCEWithLogitsLoss()
```""",
        "model_name": "ctr_model",
        "feature": "feature/double-sigmoid",
    },
    {
        "diff": """```diff
  def train_loop(dataloader):
      for batch_x, labels in dataloader:
          predictions = model(batch_x) # shape: (batch_size, 1)
-         loss = criterion(predictions, labels) # labels shape: (batch_size,)
+         loss = criterion(predictions.squeeze(), labels)
```""",
        "model_name": "regression_model",
        "feature": "feature/broadcast-silent-killer",
    },
    {
        "diff": """```diff
- val_transform = transforms.Compose([transforms.RandomCrop(32), transforms.ToTensor()])
+ val_transform = transforms.Compose([transforms.CenterCrop(32), transforms.ToTensor()])
  val_dataset = datasets.ImageFolder("data/val", transform=val_transform)
```""",
        "model_name": "vision_model",
        "feature": "feature/val-augmentation",
    },
    {
        "diff": """```diff
  criterion = nn.CrossEntropyLoss()
  for images, labels in dataloader:
      outputs = model(images)
-     loss = criterion(outputs, labels.float())
+     loss = criterion(outputs, labels.long())
```""",
        "model_name": "classifier",
        "feature": "feature/crossentropy-cast",
    }
]

OOD_FLAWS_CHOSEN_TEMPLATES = [
    "**Critical Logic Bug Detected**\n\nThe implementation contains a fatal ML anti-pattern. If you are using `BCEWithLogitsLoss`, you must not apply `torch.sigmoid()` to the output beforehand, as it applies sigmoid twice and destroys gradients. If you are using a `DataLoader` on ordered datasets, you must pass `shuffle=True` or suffer catastrophic forgetting. If your predictions and labels have mismatched dimensions like `[B, 1]` and `[B]`, PyTorch will silently broadcast them into a `[B, B]` matrix and ruin your loss. CrossEntropyLoss requires `long` targets, not `float`. Finally, never apply random augmentations to your validation set.\n\nPlease apply the requested fix immediately.",
    "**Fatal Architecture/Training Flaw**\n\nThis diff highlights a common but deadly ML bug. Applying Sigmoid before `BCEWithLogitsLoss` crushes your gradients. Forgetting `shuffle=True` on a training dataloader causes catastrophic forgetting. Failing to `squeeze()` a `[batch, 1]` tensor against a `[batch]` label causes PyTorch to silently broadcast and compute a massive, meaningless loss matrix. Passing `float()` targets to `CrossEntropyLoss` will crash. Random augmentations on the validation set will inject evaluation noise.\n\nApply the diff to resolve the issue."
]


def _build_examples(diffs, chosen_templates, category_tag):
    """Produce all combinations of diff × chosen-template."""
    examples = []
    for diff_info in diffs:
        for tmpl in chosen_templates:
            prompt = PROMPT_TEMPLATE.format(
                feature=diff_info.get("feature", "feature/unknown"),
                diff=diff_info["diff"],
            )
            chosen = tmpl.format(**diff_info)
            # SFT format combines prompt and chosen with a hard stop token
            text = f"{prompt}\n\n{chosen}<|endoftext|>"
            examples.append({"text": text})
    return examples


def generate_all() -> list[dict]:
    pool = []
    pool += _build_examples(DATA_LEAKAGE_DIFFS, DATA_LEAKAGE_CHOSEN_TEMPLATES, "data_leakage")
    pool += _build_examples(OOM_LEAK_DIFFS, OOM_LEAK_CHOSEN_TEMPLATES, "oom_leak")
    pool += _build_examples(EVAL_MODE_DIFFS, EVAL_MODE_CHOSEN_TEMPLATES, "eval_mode")
    
    # New Advanced Anti-Patterns
    pool += _build_examples(SMOTE_LEAK_DIFFS, SMOTE_LEAK_CHOSEN_TEMPLATES, "smote_leakage")
    pool += _build_examples(TIME_SERIES_LEAK_DIFFS, TIME_SERIES_LEAK_CHOSEN_TEMPLATES, "timeseries_leakage")
    pool += _build_examples(DECEPTIVE_METRIC_DIFFS, DECEPTIVE_METRIC_CHOSEN_TEMPLATES, "deceptive_metric")
    
    pool += _build_examples(PYTORCH_FLAWS_DIFFS, PYTORCH_FLAWS_CHOSEN_TEMPLATES, "pytorch_flaws")
    pool += _build_examples(ADVANCED_LEAKAGE_DIFFS, ADVANCED_LEAKAGE_CHOSEN_TEMPLATES, "advanced_leakage")
    pool += _build_examples(CLASSICAL_ML_DIFFS, CLASSICAL_ML_CHOSEN_TEMPLATES, "classical_ml")
    pool += _build_examples(NLP_FLAWS_DIFFS, NLP_FLAWS_CHOSEN_TEMPLATES, "nlp_flaws")
    pool += _build_examples(OOD_FLAWS_DIFFS, OOD_FLAWS_CHOSEN_TEMPLATES, "ood_flaws")
    
    random.shuffle(pool)
    return pool


# ════════════════════════════════════════════════════════════
#  Persist to JSONL
# ════════════════════════════════════════════════════════════

def save_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  ✓ Wrote {len(records):>3} examples → {path}")


def main():
    examples = generate_all()
    print(f"\nTotal synthetic examples: {len(examples)}")
    print(f"  • Data Leakage:    {len(DATA_LEAKAGE_DIFFS) * len(DATA_LEAKAGE_CHOSEN_TEMPLATES)}")
    print(f"  • OOM Leak:        {len(OOM_LEAK_DIFFS) * len(OOM_LEAK_CHOSEN_TEMPLATES)}")
    print(f"  • Eval Mode:       {len(EVAL_MODE_DIFFS) * len(EVAL_MODE_CHOSEN_TEMPLATES)}")
    print(f"  • SMOTE Leak:      {len(SMOTE_LEAK_DIFFS) * len(SMOTE_LEAK_CHOSEN_TEMPLATES)}")
    print(f"  • Time-Series:     {len(TIME_SERIES_LEAK_DIFFS) * len(TIME_SERIES_LEAK_CHOSEN_TEMPLATES)}")
    print(f"  • Deceptive Metr.: {len(DECEPTIVE_METRIC_DIFFS) * len(DECEPTIVE_METRIC_CHOSEN_TEMPLATES)}")
    print(f"  • PyTorch Flaws:   {len(PYTORCH_FLAWS_DIFFS) * len(PYTORCH_FLAWS_CHOSEN_TEMPLATES)}")
    print(f"  • Adv. Leakage:    {len(ADVANCED_LEAKAGE_DIFFS) * len(ADVANCED_LEAKAGE_CHOSEN_TEMPLATES)}")
    print(f"  • Classical ML:    {len(CLASSICAL_ML_DIFFS) * len(CLASSICAL_ML_CHOSEN_TEMPLATES)}")
    print(f"  • NLP Traps:       {len(NLP_FLAWS_DIFFS) * len(NLP_FLAWS_CHOSEN_TEMPLATES)}")

    # 80/20 split
    split_idx = int(len(examples) * 0.8)
    train_set = examples[:split_idx]
    valid_set = examples[split_idx:]

    save_jsonl(train_set, TRAIN_PATH)
    save_jsonl(valid_set, VALID_PATH)
    print("\nDone ✓")


if __name__ == "__main__":
    main()
