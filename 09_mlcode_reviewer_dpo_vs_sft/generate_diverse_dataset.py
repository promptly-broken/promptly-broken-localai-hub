#!/usr/bin/env python3
"""
generate_diverse_dataset.py – 1000+ Diverse SFT Examples for Senior ML PR Reviewer

Strategy: Each anti-pattern category defines parameterized CODE templates and
REVIEW templates. The generator cross-products them with random parameter
fills to produce genuinely diverse (code, review) pairs.

Output: sft_data/train.jsonl and sft_data/valid.jsonl
Each row: {"text": "<prompt>\n\n<review><|endoftext|>"}
"""

import json
import random
import itertools
from pathlib import Path

random.seed(42)

DATA_DIR = Path("sft_data")
TRAIN_PATH = DATA_DIR / "train.jsonl"
VALID_PATH = DATA_DIR / "valid.jsonl"

PROMPT_TEMPLATE = """\
You are a senior Machine Learning engineer reviewing a GitHub Pull Request.
Review the following code diff from branch `{feature}` and provide detailed, actionable feedback. Focus on ML-specific correctness, not style.

{diff}"""

BRANCH_POOL = [
    "feature/training-pipeline", "feature/model-v2", "feature/preprocessing",
    "feature/inference-service", "feature/data-pipeline", "feature/evaluation",
    "feature/hyperparams", "feature/model-training", "feature/batch-predict",
    "feature/optimization", "fix/training-bug", "fix/model-accuracy",
    "fix/data-loading", "refactor/ml-pipeline", "refactor/training-loop",
    "feature/classification", "feature/regression", "feature/segmentation",
    "feature/nlp-model", "feature/cv-model", "feature/tabular-model",
    "feature/recommender", "feature/anomaly-detection", "feature/embeddings",
    "feature/fine-tuning", "feature/transfer-learning", "feature/distillation",
]


# ════════════════════════════════════════════════════════════
#  UTILITY
# ════════════════════════════════════════════════════════════

def _generate_category(code_templates, review_templates, param_sets, n_target):
    """Generate n_target examples by sampling code×review×params combinations."""
    examples = []
    combos = list(itertools.product(
        range(len(code_templates)),
        range(len(review_templates)),
        range(len(param_sets)),
    ))
    random.shuffle(combos)

    # If we have enough combos, sample exactly n_target
    # If not, cycle through them
    for i in range(n_target):
        ci, ri, pi = combos[i % len(combos)]
        params = param_sets[pi]
        try:
            diff = code_templates[ci].format(**params)
            review = review_templates[ri].format(**params)
        except KeyError:
            continue

        feature = random.choice(BRANCH_POOL)
        prompt = PROMPT_TEMPLATE.format(feature=feature, diff=diff)
        text = f"{prompt}\n\n{review}<|endoftext|>"
        examples.append({"text": text})

    return examples


# ════════════════════════════════════════════════════════════
#  CATEGORY 1: Data Leakage (fit_transform before split)
# ════════════════════════════════════════════════════════════

CAT1_CODE = [
    # --- Template A: StandardScaler ---
    """\
```diff
+ from sklearn.preprocessing import {scaler}
+ from sklearn.model_selection import train_test_split
+
+ scaler = {scaler}()
+ {X}_scaled = scaler.fit_transform({X})
+ {X}_train, {X}_test, {y}_train, {y}_test = train_test_split(
+     {X}_scaled, {y}, test_size={test_size}, random_state=42
+ )
+ {model}.fit({X}_train, {y}_train)
+ print("Test score:", {model}.score({X}_test, {y}_test))
```""",
    # --- Template B: Pipeline-less PCA ---
    """\
```diff
+ from sklearn.decomposition import PCA
+ from sklearn.preprocessing import {scaler}
+ from sklearn.model_selection import train_test_split
+
+ scaler = {scaler}()
+ {X}_scaled = scaler.fit_transform({X})
+ pca = PCA(n_components={n_components})
+ {X}_reduced = pca.fit_transform({X}_scaled)
+ {X}_train, {X}_test, {y}_train, {y}_test = train_test_split(
+     {X}_reduced, {y}, test_size={test_size}
+ )
```""",
    # --- Template C: TF-IDF on full corpus ---
    """\
```diff
+ from sklearn.feature_extraction.text import TfidfVectorizer
+ from sklearn.model_selection import train_test_split
+
+ vectorizer = TfidfVectorizer(max_features={n_components})
+ {X}_tfidf = vectorizer.fit_transform(corpus)
+ {X}_train, {X}_test, {y}_train, {y}_test = train_test_split(
+     {X}_tfidf, {y}, test_size={test_size}
+ )
+ {model}.fit({X}_train, {y}_train)
```""",
]

CAT1_REVIEWS = [
    "**Data Leakage Detected.** You are calling `fit_transform()` on the entire dataset before splitting. The {scaler} learns statistics (mean, variance) from test samples, inflating your score. Fix: call `fit_transform` only on the training set, then `transform` on the test set.",
    "**Critical: Train/Test Contamination.** The {scaler} is fitted on all of `{X}` before the split. Test-set statistics leak into training. Split first, then fit the scaler on `{X}_train` only.",
    "**Preprocessing Leakage Bug.** `{scaler}().fit_transform({X})` computes statistics across train+test. Your reported score is unreliable. Always fit on training data only, then apply `scaler.transform()` to the test fold.",
    "This code leaks information from the test set into training. The `fit_transform` call on `{X}` includes test samples. The correct pattern: split first, then `scaler.fit({X}_train)` followed by `scaler.transform({X}_test)`.",
    "**Major Evaluation Flaw.** By fitting {scaler} before `train_test_split`, you allow the model to indirectly see test data. Your {test_size} test score will be optimistically biased. Fit all preprocessing inside the training fold.",
    "Leakage: `fit_transform` on the full dataset means the scaler's parameters encode test-set information. This is a textbook ML anti-pattern. Fix: split your data first, then fit preprocessing exclusively on training data.",
]

CAT1_PARAMS = [
    {"scaler": "StandardScaler", "X": "X", "y": "y", "model": "model", "test_size": "0.2", "n_components": "50"},
    {"scaler": "MinMaxScaler", "X": "features", "y": "labels", "model": "clf", "test_size": "0.25", "n_components": "100"},
    {"scaler": "RobustScaler", "X": "X", "y": "y", "model": "estimator", "test_size": "0.3", "n_components": "30"},
    {"scaler": "StandardScaler", "X": "data", "y": "target", "model": "classifier", "test_size": "0.15", "n_components": "64"},
    {"scaler": "MaxAbsScaler", "X": "X", "y": "y", "model": "model", "test_size": "0.2", "n_components": "20"},
    {"scaler": "Normalizer", "X": "inputs", "y": "outputs", "model": "regressor", "test_size": "0.2", "n_components": "128"},
]

# ════════════════════════════════════════════════════════════
#  CATEGORY 2: GPU Memory Leak (loss.append without .item())
# ════════════════════════════════════════════════════════════

CAT2_CODE = [
    """\
```diff
+ {model_def}
+ criterion = {loss_fn}()
+ optimizer = optim.{optimizer}({model_var}.parameters(), lr={lr})
+
+ {history} = []
+ for epoch in range({epochs}):
+     for {bx}, {by} in {loader}:
+         {bx}, {by} = {bx}.cuda(), {by}.cuda()
+         optimizer.zero_grad()
+         out = {model_var}({bx})
+         loss = criterion(out, {by})
+         loss.backward()
+         optimizer.step()
+         {history}.append(loss)
```""",
    """\
```diff
+ {history} = []
+ for step, batch in enumerate({loader}):
+     inputs, targets = batch[0].to(device), batch[1].to(device)
+     optimizer.zero_grad()
+     predictions = {model_var}(inputs)
+     loss = criterion(predictions, targets)
+     loss.backward()
+     optimizer.step()
+
+     {history}.append(loss)
+     if step % 100 == 0:
+         print(f"Step {{step}}, Loss: {{loss}}")
```""",
    """\
```diff
+ training_losses = []
+ validation_losses = []
+ for epoch in range({epochs}):
+     {model_var}.train()
+     epoch_loss = 0
+     for {bx}, {by} in {loader}:
+         optimizer.zero_grad()
+         out = {model_var}({bx}.cuda())
+         loss = criterion(out, {by}.cuda())
+         loss.backward()
+         optimizer.step()
+         training_losses.append(loss)
```""",
]

CAT2_REVIEWS = [
    "**GPU Memory Leak.** You append the raw `loss` tensor to `{history}`. This retains the entire autograd computation graph in memory. After a few epochs you will get an OOM crash. Fix: use `{history}.append(loss.item())` to detach the scalar value.",
    "**Critical OOM Bug.** Every call to `{history}.append(loss)` keeps the full backward graph alive. GPU memory will grow linearly until it crashes. Replace with `loss.item()` or `loss.detach().cpu().item()`.",
    "**Memory leak in training loop.** The `loss` tensor carries the entire computation graph. Appending it to a list prevents garbage collection. Use `loss.item()` to extract the Python float, which breaks the graph reference.",
    "Appending `loss` directly retains the computation graph. This is a classic PyTorch memory leak that will cause OOM after several epochs. The fix is trivial: `{history}.append(loss.item())`.",
    "**Autograd Graph Retention Bug.** `{history}.append(loss)` holds references to every intermediate tensor in the forward pass. Your memory consumption will grow without bound. Always call `.item()` when logging scalar losses.",
    "This training loop leaks GPU memory. The `loss` tensor is attached to the computation graph, and appending it to `{history}` prevents PyTorch from freeing that memory. Convert to a Python scalar with `loss.item()` before storing.",
]

CAT2_PARAMS = [
    {"model_def": "model = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10)).cuda()", "loss_fn": "nn.CrossEntropyLoss", "optimizer": "Adam", "lr": "0.001", "history": "epoch_losses", "epochs": "50", "bx": "batch_x", "by": "batch_y", "loader": "train_loader", "model_var": "model"},
    {"model_def": "net = ResNet18(num_classes=100).cuda()", "loss_fn": "nn.CrossEntropyLoss", "optimizer": "SGD", "lr": "0.01", "history": "loss_history", "epochs": "100", "bx": "images", "by": "labels", "loader": "dataloader", "model_var": "net"},
    {"model_def": "model = TransformerEncoder(d_model=768, nhead=12).cuda()", "loss_fn": "nn.MSELoss", "optimizer": "AdamW", "lr": "0.0001", "history": "losses", "epochs": "30", "bx": "x", "by": "y", "loader": "train_dl", "model_var": "model"},
    {"model_def": "classifier = MLP(input_dim=1024, hidden_dim=512, output_dim=5).cuda()", "loss_fn": "nn.CrossEntropyLoss", "optimizer": "Adam", "lr": "0.0005", "history": "train_losses", "epochs": "200", "bx": "features", "by": "targets", "loader": "loader", "model_var": "classifier"},
    {"model_def": "model = UNet(in_channels=3, out_channels=1).cuda()", "loss_fn": "nn.BCEWithLogitsLoss", "optimizer": "Adam", "lr": "0.001", "history": "all_losses", "epochs": "80", "bx": "imgs", "by": "masks", "loader": "seg_loader", "model_var": "model"},
]

# ════════════════════════════════════════════════════════════
#  CATEGORY 3: Missing model.eval() in Validation
# ════════════════════════════════════════════════════════════

CAT3_CODE = [
    """\
```diff
+ def validate({model_var}, {loader}):
+     total_loss = 0
+     correct = 0
+     for {bx}, {by} in {loader}:
+         {bx}, {by} = {bx}.cuda(), {by}.cuda()
+         outputs = {model_var}({bx})
+         loss = criterion(outputs, {by})
+         total_loss += loss.item()
+     return total_loss / len({loader})
```""",
    """\
```diff
+ # Validation loop
+ val_preds = []
+ with torch.no_grad():
+     for batch in val_loader:
+         inputs = batch["input_ids"].cuda()
+         logits = {model_var}(inputs)
+         val_preds.append(logits.cpu())
+ val_accuracy = compute_accuracy(val_preds, val_labels)
```""",
    """\
```diff
+ def evaluate({model_var}, test_loader, criterion):
+     test_loss = 0
+     all_preds = []
+     for {bx}, {by} in test_loader:
+         with torch.no_grad():
+             pred = {model_var}({bx}.to(device))
+             test_loss += criterion(pred, {by}.to(device)).item()
+             all_preds.extend(pred.argmax(dim=1).cpu().tolist())
+     return test_loss, all_preds
```""",
]

CAT3_REVIEWS = [
    "**Missing `model.eval()`.** Your validation function never switches the model to evaluation mode. This means `BatchNorm` layers keep using batch statistics instead of running averages, and `Dropout` layers keep randomly zeroing activations. Add `{model_var}.eval()` before the loop.",
    "**Evaluation mode not set.** Without `{model_var}.eval()`, dropout is still active and batch norm uses per-batch statistics during validation. This makes your validation metrics noisy and non-reproducible. Always call `.eval()` before inference.",
    "Bug: `{model_var}.eval()` is missing. If your model uses BatchNorm or Dropout, the validation results will be inconsistent and unreliable. Call `{model_var}.eval()` at the start and `{model_var}.train()` when resuming training.",
    "**Critical: Model still in training mode during evaluation.** Dropout will randomly drop neurons and BatchNorm will use mini-batch statistics. Your validation loss is unreliable. Fix: add `{model_var}.eval()` before the validation loop.",
    "You forgot `{model_var}.eval()`. BatchNorm and Dropout behave differently during training vs. inference. Without switching modes, your evaluation metrics are corrupted by training-time stochasticity.",
    "The model is evaluated in training mode. This is a common mistake that causes validation metrics to fluctuate unpredictably. Insert `{model_var}.eval()` before any forward pass during evaluation, and remember to call `{model_var}.train()` afterwards.",
]

CAT3_PARAMS = [
    {"model_var": "model", "loader": "val_loader", "bx": "batch_x", "by": "batch_y"},
    {"model_var": "net", "loader": "test_loader", "bx": "inputs", "by": "labels"},
    {"model_var": "classifier", "loader": "eval_loader", "bx": "x", "by": "y"},
    {"model_var": "encoder", "loader": "validation_dl", "bx": "features", "by": "targets"},
]

# ════════════════════════════════════════════════════════════
#  CATEGORY 4: Missing torch.no_grad() in Inference
# ════════════════════════════════════════════════════════════

CAT4_CODE = [
    """\
```diff
+ {model_var}.eval()
+ predictions = []
+ for {bx}, {by} in {loader}:
+     {bx} = {bx}.cuda()
+     out = {model_var}({bx})
+     predictions.append(out.cpu())
+ predictions = torch.cat(predictions)
```""",
    """\
```diff
+ {model_var}.eval()
+ total_correct = 0
+ total_samples = 0
+ for {bx}, {by} in {loader}:
+     outputs = {model_var}({bx}.to(device))
+     _, predicted = torch.max(outputs, 1)
+     total_correct += (predicted == {by}.to(device)).sum().item()
+     total_samples += {by}.size(0)
+ accuracy = total_correct / total_samples
```""",
]

CAT4_REVIEWS = [
    "**Missing `torch.no_grad()`.** You call `{model_var}.eval()` but never wrap the inference loop in `torch.no_grad()`. PyTorch is still building the computation graph for every forward pass, wasting ~30% more memory. Wrap the loop in `with torch.no_grad():`.",
    "**Unnecessary gradient computation.** Although `{model_var}.eval()` disables dropout and batch norm training behavior, it does NOT disable gradient tracking. You need `with torch.no_grad():` to prevent autograd from recording operations, reducing memory usage significantly.",
    "Good that you set `{model_var}.eval()`, but gradients are still being computed. This wastes GPU memory and slows down inference. Wrap the entire evaluation block in `with torch.no_grad():` to disable the autograd engine.",
    "You are missing `torch.no_grad()` around your inference code. Even in eval mode, PyTorch tracks gradients by default. This creates unnecessary computation graph overhead. Fix: `with torch.no_grad():` before the loop.",
    "**Performance issue.** `{model_var}.eval()` only changes layer behavior (BN, Dropout). To actually save memory during inference, you must also use `torch.no_grad()` to stop gradient tracking. This can reduce memory usage by 30-50%.",
]

CAT4_PARAMS = [
    {"model_var": "model", "loader": "val_loader", "bx": "batch_x", "by": "batch_y"},
    {"model_var": "net", "loader": "test_loader", "bx": "images", "by": "labels"},
    {"model_var": "classifier", "loader": "eval_loader", "bx": "x", "by": "y"},
    {"model_var": "backbone", "loader": "inference_dl", "bx": "inputs", "by": "targets"},
]

# ════════════════════════════════════════════════════════════
#  CATEGORY 5: Double Sigmoid (sigmoid + BCEWithLogitsLoss)
# ════════════════════════════════════════════════════════════

CAT5_CODE = [
    """\
```diff
+ class {cls}(nn.Module):
+     def __init__(self, {in_dim}):
+         super().__init__()
+         self.fc1 = nn.Linear({in_dim}, {hid_dim})
+         self.fc2 = nn.Linear({hid_dim}, 1)
+
+     def forward(self, x):
+         x = torch.relu(self.fc1(x))
+         x = self.fc2(x)
+         return torch.sigmoid(x)
+
+ {model_var} = {cls}({in_val}).cuda()
+ criterion = nn.BCEWithLogitsLoss()
```""",
    """\
```diff
+ class {cls}(nn.Module):
+     def __init__(self):
+         super().__init__()
+         self.layers = nn.Sequential(
+             nn.Linear({in_val}, {hid_dim}),
+             nn.ReLU(),
+             nn.Linear({hid_dim}, 1),
+             nn.Sigmoid()
+         )
+
+     def forward(self, x):
+         return self.layers(x)
+
+ criterion = nn.BCEWithLogitsLoss()
```""",
    """\
```diff
+ def forward(self, x):
+     x = self.encoder(x)
+     x = self.head(x)
+     return torch.sigmoid(x)
+
+ loss_fn = nn.BCEWithLogitsLoss()
+ loss = loss_fn({model_var}(inputs), targets)
```""",
]

CAT5_REVIEWS = [
    "**Double Sigmoid Bug.** Your model applies `torch.sigmoid()` in `forward()`, but `BCEWithLogitsLoss` already applies sigmoid internally. The output is squashed twice, destroying gradients. Remove the sigmoid from your model or switch to `nn.BCELoss()`.",
    "**Fatal: Sigmoid applied twice.** `BCEWithLogitsLoss` = Sigmoid + BCELoss. Your model already applies sigmoid, so the loss function applies it again. Gradients near 0 or 1 become vanishingly small. Remove `torch.sigmoid()` from the forward pass.",
    "**Gradient-killing bug.** `nn.BCEWithLogitsLoss` expects raw logits. Your model returns `sigmoid(x)`, so sigmoid is applied twice: `sigmoid(sigmoid(x))`. This flattens the gradient landscape and prevents learning. Either remove the sigmoid or use `nn.BCELoss()`.",
    "Your model output passes through sigmoid, then `BCEWithLogitsLoss` applies sigmoid again internally. Double-sigmoid compresses all outputs toward 0.5, making the model unable to express confident predictions. Remove the sigmoid from the model.",
    "**Architecture/Loss Mismatch.** `BCEWithLogitsLoss` is designed for raw logits. Since `{cls}` already applies sigmoid, you get `σ(σ(x))`. This crushes gradients and makes training nearly impossible. Fix: return raw logits from forward() and let the loss handle sigmoid.",
    "Critical bug: double activation. The sigmoid in `forward()` and the implicit sigmoid in `BCEWithLogitsLoss` are redundant. Your effective activation is `sigmoid(sigmoid(logits))`, which has near-zero gradients everywhere. Remove one of them.",
]

CAT5_PARAMS = [
    {"cls": "ClickPredictor", "in_dim": "input_dim", "hid_dim": "128", "in_val": "50", "model_var": "model"},
    {"cls": "FraudDetector", "in_dim": "n_features", "hid_dim": "256", "in_val": "100", "model_var": "detector"},
    {"cls": "BinaryClassifier", "in_dim": "d_in", "hid_dim": "64", "in_val": "768", "model_var": "clf"},
    {"cls": "ChurnModel", "in_dim": "input_size", "hid_dim": "512", "in_val": "32", "model_var": "model"},
    {"cls": "SentimentHead", "in_dim": "embed_dim", "hid_dim": "128", "in_val": "384", "model_var": "head"},
]

# ════════════════════════════════════════════════════════════
#  CATEGORY 6: Double Softmax (softmax + CrossEntropyLoss)
# ════════════════════════════════════════════════════════════

CAT6_CODE = [
    """\
```diff
+ class {cls}(nn.Module):
+     def __init__(self, {in_dim}, num_classes):
+         super().__init__()
+         self.fc = nn.Linear({in_dim}, num_classes)
+
+     def forward(self, x):
+         logits = self.fc(x)
+         return nn.Softmax(dim=1)(logits)
+
+ criterion = nn.CrossEntropyLoss()
+ loss = criterion({model_var}(inputs), labels)
```""",
    """\
```diff
+ def forward(self, x):
+     x = self.backbone(x)
+     x = self.classifier(x)
+     probs = torch.softmax(x, dim=-1)
+     return probs
+
+ criterion = nn.CrossEntropyLoss()
```""",
    """\
```diff
+     logits = self.head(features)
+     return F.softmax(logits, dim=1)
+
+ loss_fn = nn.CrossEntropyLoss()
+ pred = {model_var}(batch_x)
+ loss = loss_fn(pred, batch_y)
```""",
]

CAT6_REVIEWS = [
    "**Double Softmax Bug.** `nn.CrossEntropyLoss` applies `LogSoftmax` internally. Your model already applies `Softmax`, so softmax is computed twice: `log(softmax(softmax(x)))`. This completely flattens gradients. Remove softmax from your model.",
    "**Critical: Redundant softmax.** `CrossEntropyLoss` expects raw logits, not probabilities. Your model applies softmax before the loss, causing double application. The gradient signal is destroyed. Return raw logits from `forward()`.",
    "Your forward pass returns `softmax(logits)`, but `CrossEntropyLoss` internally applies `log_softmax`. The effective computation is `log(softmax(softmax(x)))`, which compresses gradients to near-zero. Remove the softmax from the model.",
    "**Loss function mismatch.** `nn.CrossEntropyLoss` = `LogSoftmax` + `NLLLoss`. Since your model applies softmax, the network cannot learn effectively. Fix: remove `Softmax`/`F.softmax` from the forward method and return raw logits.",
    "Bug: `CrossEntropyLoss` already normalizes logits via LogSoftmax. Applying softmax in the model means every output is double-normalized. Training will appear to converge but accuracy will plateau early. Return logits, not probabilities.",
]

CAT6_PARAMS = [
    {"cls": "Classifier", "in_dim": "input_dim", "model_var": "model"},
    {"cls": "ImageClassifier", "in_dim": "d_model", "model_var": "net"},
    {"cls": "TextClassifier", "in_dim": "hidden_size", "model_var": "classifier"},
    {"cls": "ActionPredictor", "in_dim": "state_dim", "model_var": "policy"},
]

# ════════════════════════════════════════════════════════════
#  CATEGORY 7: Broadcasting Shape Mismatch
# ════════════════════════════════════════════════════════════

CAT7_CODE = [
    """\
```diff
+ {model_var} = {cls}().cuda()
+ criterion = nn.{loss_fn}()
+ optimizer = optim.{optimizer}({model_var}.parameters(), lr={lr})
+
+ for {bx}, labels in {loader}:
+     {bx}, labels = {bx}.cuda(), labels.cuda()
+     optimizer.zero_grad()
+     # predictions shape: (batch_size, 1)
+     # labels shape: (batch_size,)
+     predictions = {model_var}({bx})
+     loss = criterion(predictions, labels)
+     loss.backward()
+     optimizer.step()
```""",
    """\
```diff
+ # Forward pass
+ output = {model_var}(features)   # shape: [{batch}, 1]
+ target = labels                  # shape: [{batch}]
+ loss = F.mse_loss(output, target)
```""",
    """\
```diff
+ preds = {model_var}(batch_x)  # [B, 1]
+ loss = nn.{loss_fn}()(preds, batch_y)  # batch_y is [B]
+ loss.backward()
```""",
]

CAT7_REVIEWS = [
    "**Silent Broadcasting Bug.** `predictions` has shape `[{batch}, 1]` and `labels` has shape `[{batch}]`. PyTorch will silently broadcast them into a `[{batch}, {batch}]` matrix, computing a completely wrong loss. Fix: `predictions.squeeze()` or `labels.unsqueeze(1)`.",
    "**Shape mismatch causes silent broadcasting.** Your model outputs `[batch_size, 1]` but labels are `[batch_size]`. PyTorch doesn't error—it broadcasts them into a `[batch_size, batch_size]` tensor. The loss is mathematically meaningless. Squeeze the predictions: `predictions.squeeze(-1)`.",
    "**Critical dimension bug.** Predictions shape `[B, 1]` vs labels shape `[B]` triggers PyTorch broadcasting. The loss becomes a `[B, B]` matrix instead of a `[B]` vector. This silently ruins training. Fix: add `.squeeze()` to align dimensions.",
    "Dangerous shape mismatch. Your predictions are `[batch, 1]` and targets are `[batch]`. Instead of erroring, PyTorch broadcasts them to `[batch, batch]`. The computed loss is nonsensical. Always ensure prediction and target shapes match exactly.",
    "**Broadcasting trap.** `nn.{loss_fn}()` does not check that predictions and labels have the same shape. With `[B, 1]` vs `[B]`, PyTorch silently expands both to `[B, B]`, multiplying your effective batch size by B. Use `predictions.squeeze(-1)` to fix.",
    "Your loss function receives mismatched shapes: predictions `[B, 1]` and labels `[B]`. PyTorch will broadcast these into a `[B, B]` matrix—every prediction is compared to every label. This is almost certainly not what you want. Squeeze predictions to `[B]`.",
]

CAT7_PARAMS = [
    {"cls": "RegressionModel", "loss_fn": "MSELoss", "optimizer": "Adam", "lr": "0.001", "model_var": "model", "bx": "batch_x", "loader": "train_loader", "batch": "batch_size"},
    {"cls": "PricePredictor", "loss_fn": "L1Loss", "optimizer": "SGD", "lr": "0.01", "model_var": "predictor", "bx": "features", "loader": "dataloader", "batch": "N"},
    {"cls": "ScoreEstimator", "loss_fn": "MSELoss", "optimizer": "AdamW", "lr": "0.0005", "model_var": "estimator", "bx": "x", "loader": "loader", "batch": "B"},
    {"cls": "AgeRegressor", "loss_fn": "SmoothL1Loss", "optimizer": "Adam", "lr": "0.002", "model_var": "regressor", "bx": "inputs", "loader": "train_dl", "batch": "batch_size"},
]

# ════════════════════════════════════════════════════════════
#  CATEGORY 8: Sequential DataLoader (missing shuffle=True)
# ════════════════════════════════════════════════════════════

CAT8_CODE = [
    """\
```diff
+ transform = transforms.Compose([transforms.Resize(({img_size}, {img_size})), transforms.ToTensor()])
+ train_dataset = datasets.ImageFolder("{data_dir}/train", transform=transform)
+ train_loader = DataLoader(train_dataset, batch_size={bs}, num_workers={nw}, pin_memory=True)
```""",
    """\
```diff
+ dataset = {dataset_cls}("{data_dir}", train=True, download=True, transform=transform)
+ loader = DataLoader(dataset, batch_size={bs}, num_workers={nw})
```""",
    """\
```diff
+ train_ds = CustomDataset(train_df, tokenizer, max_len={max_len})
+ train_loader = DataLoader(train_ds, batch_size={bs}, num_workers={nw}, pin_memory=True)
```""",
]

CAT8_REVIEWS = [
    "**Missing `shuffle=True` on training DataLoader.** Without shuffling, the model sees examples in the same fixed order every epoch. For ordered datasets (like ImageFolder which sorts by class), this means entire batches contain the same class, causing catastrophic forgetting. Add `shuffle=True`.",
    "**DataLoader not shuffled.** The training DataLoader lacks `shuffle=True`. If the dataset has any inherent ordering (by class, by time, by source), the model will learn spurious correlations with position. Always shuffle training data.",
    "Bug: `shuffle=True` is missing from the training DataLoader. This means the model processes data in the same deterministic order every epoch. For classification tasks, this can cause entire batches to be from a single class, severely hurting convergence.",
    "**Sequential batching detected.** Your DataLoader defaults to `shuffle=False`. With ordered datasets, the optimizer will see class-homogeneous batches, causing gradient estimates to be heavily biased. Fix: `DataLoader(..., shuffle=True)`.",
    "The training DataLoader is missing `shuffle=True`. Without shuffling, gradient updates are correlated across consecutive batches, leading to poor convergence and potential overfitting to batch ordering. Always shuffle your training data.",
]

CAT8_PARAMS = [
    {"img_size": "224", "data_dir": "data", "bs": "64", "nw": "4", "dataset_cls": "datasets.CIFAR10", "max_len": "512"},
    {"img_size": "256", "data_dir": "imagenet", "bs": "32", "nw": "8", "dataset_cls": "datasets.MNIST", "max_len": "128"},
    {"img_size": "384", "data_dir": "images", "bs": "16", "nw": "4", "dataset_cls": "datasets.FashionMNIST", "max_len": "256"},
    {"img_size": "224", "data_dir": "dataset", "bs": "128", "nw": "2", "dataset_cls": "datasets.SVHN", "max_len": "1024"},
]

# ════════════════════════════════════════════════════════════
#  CATEGORY 9: SMOTE before Cross-Validation
# ════════════════════════════════════════════════════════════

CAT9_CODE = [
    """\
```diff
+ from imblearn.over_sampling import SMOTE
+ from sklearn.model_selection import cross_val_score
+
+ smote = SMOTE(random_state=42)
+ {X}_res, {y}_res = smote.fit_resample({X}, {y})
+
+ {model_var} = {clf}({clf_params})
+ scores = cross_val_score({model_var}, {X}_res, {y}_res, cv={cv}, scoring='{metric}')
+ print(f"CV {metric}: {{scores.mean():.4f}}")
```""",
    """\
```diff
+ from imblearn.over_sampling import {sampler}
+ from sklearn.model_selection import StratifiedKFold
+
+ sampler = {sampler}(random_state=42)
+ {X}_balanced, {y}_balanced = sampler.fit_resample({X}, {y})
+
+ skf = StratifiedKFold(n_splits={cv})
+ for train_idx, val_idx in skf.split({X}_balanced, {y}_balanced):
+     {model_var}.fit({X}_balanced[train_idx], {y}_balanced[train_idx])
```""",
]

CAT9_REVIEWS = [
    "**Data leakage via SMOTE.** You apply SMOTE to the entire dataset before cross-validation. Synthetic samples generated from validation-fold instances leak into training folds, inflating your {metric} score. Apply SMOTE inside the CV loop using `imblearn.pipeline.Pipeline`.",
    "**Cross-validation leakage.** Oversampling with {sampler} before splitting means synthetic minority samples are generated from data that ends up in the validation fold. Your CV score is unreliable. Resample inside each fold.",
    "**Critical evaluation flaw.** SMOTE generates synthetic samples by interpolating between existing points. If you oversample before CV, validation folds contain synthetic copies of their own real samples. The reported {metric} will be overly optimistic.",
    "Bug: SMOTE applied before cross-validation. Synthetic samples created from test-fold points contaminate the training fold. Your {cv}-fold CV score is inflated. Fix: use `imblearn.pipeline.make_pipeline({sampler}(), {clf}())` inside the cross-validation.",
    "**Oversampling leakage.** By resampling before splitting, the validation set contains synthetic points derived from real points in the same fold. This creates information leakage. Always apply resampling within each cross-validation fold.",
]

CAT9_PARAMS = [
    {"X": "X", "y": "y", "model_var": "model", "clf": "RandomForestClassifier", "clf_params": "n_estimators=100", "cv": "5", "metric": "f1", "sampler": "SMOTE"},
    {"X": "features", "y": "labels", "model_var": "clf", "clf": "GradientBoostingClassifier", "clf_params": "", "cv": "10", "metric": "roc_auc", "sampler": "ADASYN"},
    {"X": "X", "y": "y", "model_var": "estimator", "clf": "SVC", "clf_params": "kernel='rbf'", "cv": "5", "metric": "f1_macro", "sampler": "BorderlineSMOTE"},
    {"X": "data", "y": "target", "model_var": "model", "clf": "LogisticRegression", "clf_params": "max_iter=1000", "cv": "3", "metric": "precision", "sampler": "SMOTE"},
]

# ════════════════════════════════════════════════════════════
#  CATEGORY 10: Time-Series Random Split
# ════════════════════════════════════════════════════════════

CAT10_CODE = [
    """\
```diff
+ df = pd.read_csv("{data_file}", parse_dates=["{date_col}"]).sort_values("{date_col}")
+ X = df[{feature_list}]
+ y = df["{target}"]
+
+ X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=42)
+ {model_var} = {model_cls}({model_params})
+ {model_var}.fit(X_train, y_train)
```""",
    """\
```diff
+ from sklearn.model_selection import train_test_split
+
+ # Daily {target} data sorted by date
+ X_train, X_test, y_train, y_test = train_test_split(
+     features, targets, test_size={test_size}, random_state=42, shuffle=True
+ )
+ {model_var} = {model_cls}({model_params})
+ {model_var}.fit(X_train, y_train)
```""",
]

CAT10_REVIEWS = [
    "**Temporal data leakage.** You use `train_test_split` with random shuffling on time-series data. This means the model trains on future data to predict the past. Use `TimeSeriesSplit` or split sequentially with `shuffle=False`.",
    "**Time-series split violation.** Random splitting destroys temporal ordering. The model sees Friday's data when predicting Monday's {target}. This is look-ahead bias. Split chronologically: train on the first N% of dates, test on the rest.",
    "**Critical: Random shuffle on temporal data.** `train_test_split(shuffle=True)` on time-series data allows future information to leak into training. Your test metrics will be wildly optimistic. Use chronological splitting or `TimeSeriesSplit`.",
    "**Look-ahead bias.** By randomly shuffling {date_col}-sorted data, your model gets access to future values during training. Real-world performance will be far worse than reported. Split by time: `train = df[df['{date_col}'] < cutoff]`.",
    "Bug: `train_test_split` with `random_state=42` randomly shuffles time-ordered data. The model can use future observations to predict past ones—this is not achievable in production. Always preserve temporal ordering in your splits.",
]

CAT10_PARAMS = [
    {"data_file": "daily_stock_prices.csv", "date_col": "Date", "target": "Close", "feature_list": '["Open", "High", "Low", "Volume"]', "test_size": "0.2", "model_var": "model", "model_cls": "XGBRegressor", "model_params": "n_estimators=200"},
    {"data_file": "energy_consumption.csv", "date_col": "timestamp", "target": "usage_kwh", "feature_list": '["temp", "humidity", "hour", "day_of_week"]', "test_size": "0.25", "model_var": "regressor", "model_cls": "RandomForestRegressor", "model_params": "n_estimators=100"},
    {"data_file": "sales_data.csv", "date_col": "order_date", "target": "revenue", "feature_list": '["price", "discount", "quantity"]', "test_size": "0.2", "model_var": "model", "model_cls": "LGBMRegressor", "model_params": "num_leaves=31"},
    {"data_file": "web_traffic.csv", "date_col": "date", "target": "visits", "feature_list": '["weekday", "is_holiday", "campaign_active"]', "test_size": "0.3", "model_var": "predictor", "model_cls": "LinearRegression", "model_params": ""},
]

# ════════════════════════════════════════════════════════════
#  CATEGORY 11: Accuracy on Imbalanced Data
# ════════════════════════════════════════════════════════════

CAT11_CODE = [
    """\
```diff
+ # Dataset: {n_total} samples, only {n_pos} are positive ({ratio}% positive rate)
+ X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, stratify=y)
+
+ {model_var} = {clf}({clf_params})
+ {model_var}.fit(X_train, y_train)
+
+ predictions = {model_var}.predict(X_test)
+ acc = accuracy_score(y_test, predictions)
+ print(f"Model Accuracy: {{acc * 100:.2f}}%")
+
+ if acc > {threshold}:
+     print("Model is performing excellent! Deploying to production.")
```""",
    """\
```diff
+ # Evaluate {task} model
+ y_pred = {model_var}.predict(X_test)
+ print(f"Accuracy: {{accuracy_score(y_test, y_pred):.4f}}")
+ # 99.5% accuracy achieved! Ship it.
```""",
]

CAT11_REVIEWS = [
    "**Deceptive metric.** With only {ratio}% positive rate, a dummy classifier that always predicts the majority class achieves ~{neg_acc}% accuracy. Your {threshold} threshold is meaningless. Use F1-score, Precision-Recall AUC, or Balanced Accuracy instead.",
    "**Accuracy is misleading on imbalanced data.** Your dataset has a {ratio}% positive rate. A model that predicts 0 for everything gets {neg_acc}% accuracy. Report precision, recall, F1-score, and confusion matrix instead.",
    "**Critical evaluation error.** Accuracy on highly imbalanced data ({ratio}% positives) is not informative. Your model could be predicting all negatives and still appear to perform well. Use `classification_report` or `average_precision_score` for a realistic assessment.",
    "**Wrong metric for imbalanced {task}.** With {n_pos}/{n_total} positive samples, accuracy is dominated by the majority class. A trivial baseline gets {neg_acc}%. Switch to F1, AUPRC, or at minimum report the confusion matrix.",
    "Your deployment gate (`acc > {threshold}`) is fundamentally flawed for this class distribution. With {ratio}% positives, even random guessing clears that bar. Use metrics that account for class imbalance: F1, MCC, or balanced accuracy.",
]

CAT11_PARAMS = [
    {"n_total": "10000", "n_pos": "100", "ratio": "1", "neg_acc": "99", "test_size": "0.2", "model_var": "model", "clf": "LogisticRegression", "clf_params": "class_weight=None", "threshold": "0.95", "task": "fraud detection"},
    {"n_total": "50000", "n_pos": "250", "ratio": "0.5", "neg_acc": "99.5", "test_size": "0.2", "model_var": "clf", "clf": "RandomForestClassifier", "clf_params": "n_estimators=100", "threshold": "0.98", "task": "anomaly detection"},
    {"n_total": "100000", "n_pos": "500", "ratio": "0.5", "neg_acc": "99.5", "test_size": "0.25", "model_var": "detector", "clf": "GradientBoostingClassifier", "clf_params": "", "threshold": "0.99", "task": "rare event detection"},
    {"n_total": "20000", "n_pos": "400", "ratio": "2", "neg_acc": "98", "test_size": "0.2", "model_var": "model", "clf": "SVC", "clf_params": "", "threshold": "0.95", "task": "disease screening"},
]

# ════════════════════════════════════════════════════════════
#  CATEGORY 12: Augmentation on Validation/Test Set
# ════════════════════════════════════════════════════════════

CAT12_CODE = [
    """\
```diff
+ transform = transforms.Compose([
+     transforms.Resize(({img_size}, {img_size})),
+     transforms.RandomHorizontalFlip(),
+     transforms.RandomRotation({rotation}),
+     transforms.ColorJitter(brightness={jitter}),
+     transforms.ToTensor(),
+     transforms.Normalize(mean={mean}, std={std})
+ ])
+
+ train_dataset = datasets.ImageFolder("{data_dir}/train", transform=transform)
+ val_dataset = datasets.ImageFolder("{data_dir}/val", transform=transform)
```""",
    """\
```diff
+ aug_transform = transforms.Compose([
+     transforms.RandomResizedCrop({img_size}),
+     transforms.RandomHorizontalFlip(p=0.5),
+     transforms.RandomAffine(degrees={rotation}, translate=(0.1, 0.1)),
+     transforms.ToTensor(),
+ ])
+
+ train_set = ImageDataset(train_paths, transform=aug_transform)
+ test_set = ImageDataset(test_paths, transform=aug_transform)
```""",
]

CAT12_REVIEWS = [
    "**Augmentation applied to validation set.** The same `transform` with `RandomHorizontalFlip`, `RandomRotation`, and `ColorJitter` is used for both train and val datasets. Random augmentations on the validation set make evaluation metrics noisy and non-reproducible. Use a separate deterministic transform for validation.",
    "**Evaluation noise from augmentations.** Your validation/test set uses the same random augmentations as training. Each evaluation pass will produce different results. Create a separate `val_transform` with only `Resize`, `ToTensor`, and `Normalize`—no random transforms.",
    "**Non-deterministic evaluation.** Applying `RandomHorizontalFlip` and `RandomRotation({rotation})` to the validation set means your metrics change every time you run evaluation. Validation transforms should be deterministic: resize, center crop, normalize only.",
    "Bug: train and val share the same random-augmentation pipeline. This corrupts your validation metrics with randomness. Define a separate `val_transform = transforms.Compose([Resize, CenterCrop, ToTensor, Normalize])` for evaluation.",
    "**Data augmentation leak into evaluation.** Random transforms on validation data make it impossible to get consistent metrics. Every eval run gives different numbers. Validation transforms should only include deterministic preprocessing (resize, normalize).",
]

CAT12_PARAMS = [
    {"img_size": "224", "data_dir": "data", "rotation": "15", "jitter": "0.3", "mean": "[0.485, 0.456, 0.406]", "std": "[0.229, 0.224, 0.225]"},
    {"img_size": "256", "data_dir": "images", "rotation": "30", "jitter": "0.2", "mean": "[0.5, 0.5, 0.5]", "std": "[0.5, 0.5, 0.5]"},
    {"img_size": "384", "data_dir": "imagenet", "rotation": "10", "jitter": "0.4", "mean": "[0.485, 0.456, 0.406]", "std": "[0.229, 0.224, 0.225]"},
]

# ════════════════════════════════════════════════════════════
#  CATEGORY 13: CrossEntropyLoss with Wrong Target Type
# ════════════════════════════════════════════════════════════

CAT13_CODE = [
    """\
```diff
+ criterion = nn.CrossEntropyLoss()
+
+ for {bx}, {by} in {loader}:
+     {bx} = {bx}.cuda()
+     {by} = {by}.float().cuda()
+     logits = {model_var}({bx})
+     loss = criterion(logits, {by})
```""",
    """\
```diff
+ labels = torch.tensor([0, 1, 2, 0, 1], dtype=torch.float32)
+ outputs = {model_var}(inputs)
+ loss = nn.CrossEntropyLoss()(outputs, labels)
```""",
    """\
```diff
+ targets = batch["{target_key}"].to(device)
+ outputs = {model_var}(inputs)
+ loss = nn.CrossEntropyLoss()(outputs.view(-1, vocab_size), targets.float().view(-1))
```""",
]

CAT13_REVIEWS = [
    "**CrossEntropyLoss target type error.** `nn.CrossEntropyLoss` requires targets of type `torch.long` (integer class indices). You cast `{by}` to `.float()`, which will crash or produce incorrect results. Use `{by}.long().cuda()` instead.",
    "**Integer targets required.** `CrossEntropyLoss` expects `LongTensor` class indices, not floats. You pass `{by}.float()` as the target. Remove the `.float()` cast and use `.long()` if needed.",
    "**CrossEntropyLoss + float targets = crash.** `nn.CrossEntropyLoss` internally uses `log_softmax` + `nll_loss`, which requires 64-bit integer class indices. Passing float targets raises a RuntimeError. Cast targets to `.long()`.",
    "**Wrong tensor type for classification loss.** CrossEntropyLoss targets must be integer class indices (dtype `torch.long`). Your code converts them to `float`, which is incompatible. Fix: `{by} = {by}.long().cuda()`.",
    "**Target tensor type mismatch.** PyTorch's `CrossEntropyLoss` requires integer class labels, not floating-point values. The `.float()` cast on your targets is incorrect. Replace with `.long()` to provide proper class indices.",
]

CAT13_PARAMS = [
    {"bx": "batch_x", "by": "batch_y", "loader": "train_loader", "model_var": "model", "target_key": "labels"},
    {"bx": "images", "by": "labels", "loader": "dataloader", "model_var": "classifier", "target_key": "class_id"},
    {"bx": "inputs", "by": "targets", "loader": "loader", "model_var": "net", "target_key": "target"},
    {"bx": "x", "by": "y", "loader": "train_dl", "model_var": "model", "target_key": "label"},
]

# ════════════════════════════════════════════════════════════
#  CATEGORY 14: .view() on Permuted Tensor
# ════════════════════════════════════════════════════════════

CAT14_CODE = [
    """\
```diff
+ x = x.permute({perm})  # shape [{shape_after}]
+ x = x.view(batch_size, -1)
+ out = self.fc(x)
```""",
    """\
```diff
+ # Reshape for attention
+ q = q.permute(0, 2, 1, 3)  # [B, heads, seq, d_k]
+ q = q.view(batch_size, -1, {dim})
```""",
    """\
```diff
+ features = features.transpose({d1}, {d2})
+ flat = features.view(features.size(0), -1)
```""",
]

CAT14_REVIEWS = [
    "**Dangerous `.view()` after permute.** `.view()` requires contiguous memory, but `.permute()` creates a non-contiguous tensor. If PyTorch doesn't error, it silently scrambles your data. Use `.reshape()` or `.contiguous().view()` instead.",
    "**Memory layout bug.** After `.permute({perm})`, the tensor is no longer contiguous in memory. `.view()` on a non-contiguous tensor either crashes or silently reinterprets the data incorrectly. Replace with `.reshape()` which handles non-contiguous tensors.",
    "**Silent data corruption.** `.permute()` changes the logical layout without moving data in memory. A subsequent `.view()` interprets the old physical layout as if it matched the new logical shape, scrambling values. Use `.contiguous().view()` or `.reshape()`.",
    "Bug: `.view()` after `.permute()` or `.transpose()` is unsafe. The tensor is non-contiguous after permutation, so `.view()` may silently produce garbage. Use `.reshape()` which works correctly on non-contiguous tensors.",
    "**Tensor layout error.** After calling `.permute({perm})`, memory layout doesn't match logical shape. `.view()` assumes contiguous memory and will produce incorrect results. Fix: replace `.view()` with `.reshape()` or call `.contiguous()` first.",
]

CAT14_PARAMS = [
    {"perm": "0, 2, 1", "shape_after": "batch, features, seq_len", "dim": "768", "d1": "1", "d2": "2"},
    {"perm": "0, 3, 1, 2", "shape_after": "batch, channels, height, width", "dim": "512", "d1": "2", "d2": "3"},
    {"perm": "1, 0, 2", "shape_after": "seq_len, batch, d_model", "dim": "256", "d1": "0", "d2": "1"},
]

# ════════════════════════════════════════════════════════════
#  CATEGORY 15: Missing optimizer.zero_grad()
# ════════════════════════════════════════════════════════════

CAT15_CODE = [
    """\
```diff
+ for epoch in range({epochs}):
+     for {bx}, {by} in {loader}:
+         {bx}, {by} = {bx}.cuda(), {by}.cuda()
+         outputs = {model_var}({bx})
+         loss = criterion(outputs, {by})
+         loss.backward()
+         optimizer.step()
```""",
    """\
```diff
+ for step, (inputs, targets) in enumerate({loader}):
+     pred = {model_var}(inputs.to(device))
+     loss = loss_fn(pred, targets.to(device))
+     loss.backward()
+     optimizer.step()
+     if step % 100 == 0:
+         print(f"Step {{step}}: loss={{loss.item():.4f}}")
```""",
]

CAT15_REVIEWS = [
    "**Missing `optimizer.zero_grad()`.** Without zeroing gradients before each backward pass, gradients accumulate across iterations. The model's weights will explode or diverge. Add `optimizer.zero_grad()` before `loss.backward()`.",
    "**Gradient accumulation bug.** You never call `optimizer.zero_grad()`. PyTorch accumulates gradients by default, so each `loss.backward()` adds to the existing gradients. After a few steps, the effective learning rate is enormous. Add `optimizer.zero_grad()` at the start of each iteration.",
    "**Critical training bug.** `optimizer.zero_grad()` is missing from the training loop. Gradients from previous batches are never cleared, causing them to accumulate indefinitely. The model will fail to converge. Zero gradients before each backward pass.",
    "Bug: no `optimizer.zero_grad()` call. PyTorch does not automatically reset gradients. Without zeroing, gradients grow unboundedly across steps. The optimizer takes increasingly large (and incorrect) steps. Fix: add `optimizer.zero_grad()` before `loss.backward()`.",
    "**Unbounded gradient accumulation.** Each `loss.backward()` adds new gradients on top of the old ones. Without `optimizer.zero_grad()`, the effective gradient at step N is the sum of all N gradients. This causes training instability or divergence.",
]

CAT15_PARAMS = [
    {"epochs": "50", "bx": "batch_x", "by": "batch_y", "loader": "train_loader", "model_var": "model"},
    {"epochs": "100", "bx": "images", "by": "labels", "loader": "dataloader", "model_var": "net"},
    {"epochs": "30", "bx": "x", "by": "y", "loader": "loader", "model_var": "classifier"},
    {"epochs": "200", "bx": "features", "by": "targets", "loader": "train_dl", "model_var": "encoder"},
]

# ════════════════════════════════════════════════════════════
#  CATEGORY 16: In-place Operations Breaking Autograd
# ════════════════════════════════════════════════════════════

CAT16_CODE = [
    """\
```diff
+ def forward(self, x):
+     x = self.linear1(x)
+     x.relu_()
+     x = self.linear2(x)
+     return x
```""",
    """\
```diff
+ hidden = self.encoder(input_ids)
+ hidden.add_(self.bias)
+ output = self.decoder(hidden)
```""",
    """\
```diff
+ features = backbone(images)
+ features.mul_(scale_factor)
+ logits = head(features)
+ loss = criterion(logits, labels)
+ loss.backward()
```""",
]

CAT16_REVIEWS = [
    "**In-place operation breaks autograd.** `relu_()` (note the underscore) modifies the tensor in-place. This can corrupt the computation graph, causing incorrect gradients or RuntimeErrors during `backward()`. Use `torch.relu(x)` or `F.relu(x)` instead.",
    "**Dangerous in-place operation.** In-place operations like `.add_()`, `.mul_()`, or `.relu_()` modify tensors that may be needed for gradient computation. This can silently produce wrong gradients. Use out-of-place versions: `x = x + bias` instead of `x.add_(bias)`.",
    "**Autograd compatibility issue.** In-place operations (marked with `_` suffix) can invalidate the computation graph. If `x` is needed for backward, modifying it in-place produces incorrect gradients. Replace `x.relu_()` with `x = F.relu(x)`.",
    "Bug: in-place operations (`relu_()`, `add_()`, `mul_()`) modify tensors that autograd may need for computing gradients. This can cause `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`. Use functional equivalents.",
    "**Silent gradient corruption.** In-place ops like `.mul_(scale_factor)` overwrite tensor data that the backward pass needs. Gradients may be silently wrong without any error. Always prefer out-of-place operations during training.",
]

CAT16_PARAMS = [
    {},  # No parameterization needed—templates are varied enough
]

# ════════════════════════════════════════════════════════════
#  CATEGORY 17: Not Detaching Hidden States in RNNs
# ════════════════════════════════════════════════════════════

CAT17_CODE = [
    """\
```diff
+ hidden = {model_var}.init_hidden(batch_size)
+ for epoch in range({epochs}):
+     for {bx}, {by} in {loader}:
+         output, hidden = {model_var}({bx}, hidden)
+         loss = criterion(output, {by})
+         loss.backward()
+         optimizer.step()
+         optimizer.zero_grad()
```""",
    """\
```diff
+ h_0 = torch.zeros({n_layers}, batch_size, {hidden_size}).to(device)
+ c_0 = torch.zeros({n_layers}, batch_size, {hidden_size}).to(device)
+
+ for seq_batch, targets in {loader}:
+     output, (h_0, c_0) = {model_var}(seq_batch, (h_0, c_0))
+     loss = criterion(output, targets)
+     loss.backward()
+     optimizer.step()
```""",
]

CAT17_REVIEWS = [
    "**Hidden state not detached.** You reuse `hidden` across batches without detaching it. This forces backpropagation through the entire sequence history (unbounded BPTT), causing memory to grow linearly with training steps. Fix: `hidden = hidden.detach()` (or `tuple(h.detach() for h in hidden)` for LSTM) at the start of each batch.",
    "**BPTT memory leak.** The hidden state carries the computation graph from all previous batches. Without `hidden.detach()`, each `backward()` tries to backpropagate through the entire training history. Memory will explode. Detach the hidden state between batches.",
    "**Unbounded backpropagation through time.** By passing `hidden` from one batch to the next without `.detach()`, you create a computation graph that spans the entire dataset. This causes OOM. Truncate BPTT by detaching: `hidden = hidden.detach()`.",
    "Bug: RNN hidden states must be detached between batches to truncate backpropagation through time. Without detaching, the backward pass tries to compute gradients all the way back to the first batch. Fix: `h_0 = h_0.detach(); c_0 = c_0.detach()`.",
    "**Memory leak via RNN hidden state.** Each forward pass extends the computation graph through the hidden state. Without `hidden.detach()`, GPU memory grows with every batch. Always detach hidden states at batch boundaries.",
]

CAT17_PARAMS = [
    {"model_var": "rnn", "epochs": "50", "bx": "seq", "by": "targets", "loader": "train_loader", "n_layers": "2", "hidden_size": "256"},
    {"model_var": "lstm", "epochs": "30", "bx": "x_seq", "by": "y_seq", "loader": "dataloader", "n_layers": "3", "hidden_size": "512"},
    {"model_var": "gru_model", "epochs": "100", "bx": "batch", "by": "labels", "loader": "seq_loader", "n_layers": "1", "hidden_size": "128"},
]

# ════════════════════════════════════════════════════════════
#  CATEGORY 18: Target Leakage via Feature Engineering
# ════════════════════════════════════════════════════════════

CAT18_CODE = [
    """\
```diff
+ # Feature engineering
+ df["{leak_feat}"] = df.groupby("{group_col}")["{target}"].transform("mean")
+
+ X = df.drop("{target}", axis=1)
+ y = df["{target}"]
+ X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```""",
    """\
```diff
+ # Add aggregate features
+ df["{leak_feat}"] = df["{target}"].rolling(window={window}).mean()
+ df["{leak_feat2}"] = df["{target}"].shift(-1)
+
+ features = df.drop(columns=["{target}"])
+ target = df["{target}"]
```""",
    """\
```diff
+ # Encode category with target statistics
+ means = df.groupby("{group_col}")["{target}"].mean()
+ df["{leak_feat}"] = df["{group_col}"].map(means)
+
+ model.fit(df.drop("{target}", axis=1), df["{target}"])
```""",
]

CAT18_REVIEWS = [
    "**Target leakage.** `{leak_feat}` is computed directly from `{target}`. This feature encodes the target variable itself, so the model trivially achieves near-perfect accuracy by reading the leaked feature. Remove target-derived features from your input.",
    "**Critical: feature derived from target.** The feature `{leak_feat}` is calculated using `{target}` values. This is textbook target leakage—the model will perform perfectly in training but fail completely in production where the target is unknown.",
    "**Information leakage via feature engineering.** You create `{leak_feat}` from the `{target}` column. Since this feature perfectly correlates with the target, the model memorizes the mapping instead of learning real patterns. Drop this feature.",
    "Bug: `{leak_feat}` is derived from the target variable `{target}`. When deployed, you won't have access to the target at prediction time, so this feature is unavailable. Your model's real-world performance will collapse. Remove all target-derived features.",
    "**Target variable leak.** Computing group means of `{target}` and using them as a feature gives the model direct access to the answer. Cross-validated scores are artificially inflated. Remove `{leak_feat}` or compute it only within the training fold.",
]

CAT18_PARAMS = [
    {"leak_feat": "avg_price_by_category", "group_col": "category", "target": "price", "leak_feat2": "next_price", "window": "3"},
    {"leak_feat": "mean_target_by_user", "group_col": "user_id", "target": "purchased", "leak_feat2": "future_purchase", "window": "5"},
    {"leak_feat": "avg_score_by_region", "group_col": "region", "target": "score", "leak_feat2": "next_score", "window": "7"},
    {"leak_feat": "claim_rate_by_group", "group_col": "age_group", "target": "is_fraud", "leak_feat2": "future_claim", "window": "10"},
]

# ════════════════════════════════════════════════════════════
#  CATEGORY 19: Not Scaling Features for Distance-Based Models
# ════════════════════════════════════════════════════════════

CAT19_CODE = [
    """\
```diff
+ from sklearn.{model_module} import {model_cls}
+ from sklearn.model_selection import train_test_split
+
+ X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
+
+ # Feature ranges: {feat1} [0-1], {feat2} [0-100000]
+ {model_var} = {model_cls}({model_params})
+ {model_var}.fit(X_train, y_train)
+ print("Score:", {model_var}.score(X_test, y_test))
```""",
    """\
```diff
+ # Features have very different scales
+ # {feat1}: 0-1, {feat2}: 0-{max_val}
+ model = {model_cls}({model_params})
+ model.fit(X_train, y_train)
```""",
]

CAT19_REVIEWS = [
    "**Features not scaled for distance-based model.** `{model_cls}` uses distance metrics internally. Feature `{feat2}` (range 0-{max_val}) will dominate over `{feat1}` (range 0-1) because of its larger scale. Apply `StandardScaler` or `MinMaxScaler` before fitting.",
    "**Scaling required for {model_cls}.** Distance-based algorithms are sensitive to feature magnitudes. Without scaling, `{feat2}` will completely dominate the distance calculations, making `{feat1}` effectively invisible. Normalize or standardize all features first.",
    "**Missing feature scaling.** `{model_cls}` relies on Euclidean distance (or similar). With `{feat1}` in [0, 1] and `{feat2}` in [0, {max_val}], the model treats `{feat2}` as orders of magnitude more important. Always scale features for distance-based methods.",
    "Bug: `{model_cls}` is scale-sensitive, but no feature scaling is applied. The large range of `{feat2}` will dominate predictions. Fix: add `StandardScaler` in a pipeline before the model.",
    "**Feature dominance issue.** Without normalization, `{model_cls}` is biased toward high-magnitude features. `{feat2}` has ~{max_val}x the range of `{feat1}`, so it controls all distance calculations. Use `sklearn.preprocessing.StandardScaler`.",
]

CAT19_PARAMS = [
    {"model_module": "neighbors", "model_cls": "KNeighborsClassifier", "model_params": "n_neighbors=5", "model_var": "knn", "feat1": "age_normalized", "feat2": "income", "max_val": "100000"},
    {"model_module": "svm", "model_cls": "SVC", "model_params": "kernel='rbf'", "model_var": "svm", "feat1": "feature_a", "feat2": "feature_b", "max_val": "50000"},
    {"model_module": "neighbors", "model_cls": "KNeighborsRegressor", "model_params": "n_neighbors=10", "model_var": "knn_reg", "feat1": "ratio", "feat2": "count", "max_val": "10000"},
    {"model_module": "cluster", "model_cls": "KMeans", "model_params": "n_clusters=5", "model_var": "kmeans", "feat1": "normalized_score", "feat2": "total_amount", "max_val": "500000"},
]

# ════════════════════════════════════════════════════════════
#  CATEGORY 20: Tokenizer/Vocab Fitted on Full Dataset
# ════════════════════════════════════════════════════════════

CAT20_CODE = [
    """\
```diff
+ from sklearn.feature_extraction.text import {vectorizer}
+ from sklearn.model_selection import train_test_split
+
+ vec = {vectorizer}(max_features={max_feat})
+ X_vec = vec.fit_transform(corpus)
+
+ X_train, X_test, y_train, y_test = train_test_split(X_vec, labels, test_size=0.2)
+ {model_var}.fit(X_train, y_train)
```""",
    """\
```diff
+ tokenizer = Tokenizer(num_words={max_feat})
+ tokenizer.fit_on_texts(all_texts)  # Includes test set!
+ sequences = tokenizer.texts_to_sequences(all_texts)
+
+ X_train, X_test = sequences[:split_idx], sequences[split_idx:]
```""",
]

CAT20_REVIEWS = [
    "**Vocabulary leakage.** The {vectorizer} is fitted on the entire corpus (train + test). Test-set vocabulary leaks into the feature space. Fit the vectorizer on training data only, then `transform` the test set.",
    "**NLP data leakage.** `fit_transform` on the full corpus means the vocabulary includes test-set terms. Words unique to the test set get dedicated features, inflating your model's apparent performance. Fit only on training data.",
    "**Text preprocessing leakage.** By fitting the tokenizer/vectorizer on `all_texts` before splitting, you allow test-set vocabulary to influence the feature space. IDF weights also incorporate test document frequencies. Split first, then fit on training text only.",
    "Bug: the {vectorizer} learns its vocabulary from the entire dataset. At test time in production, new words won't have features. Your offline metrics are optimistically biased because test words already exist in the vocabulary. Fit on training data only.",
    "**Critical: tokenizer fitted on test data.** The vocabulary and IDF statistics are computed from train+test combined. This leaks test-set information into the model. Always fit NLP preprocessors exclusively on the training split.",
]

CAT20_PARAMS = [
    {"vectorizer": "TfidfVectorizer", "max_feat": "10000", "model_var": "classifier"},
    {"vectorizer": "CountVectorizer", "max_feat": "5000", "model_var": "model"},
    {"vectorizer": "HashingVectorizer", "max_feat": "50000", "model_var": "clf"},
    {"vectorizer": "TfidfVectorizer", "max_feat": "20000", "model_var": "nb_model"},
]


# ════════════════════════════════════════════════════════════
#  ASSEMBLY & GENERATION
# ════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════
#  CATEGORY 21: CLEAN CODE — No bugs! (Negative examples)
#  These teach the model that not every code snippet is buggy.
# ════════════════════════════════════════════════════════════

CAT21_CODE = [
    # --- Correct BCEWithLogitsLoss usage (NO sigmoid in model) ---
    """\
```diff
+ class BinaryClassifier(nn.Module):
+     def __init__(self, input_dim):
+         super().__init__()
+         self.fc1 = nn.Linear(input_dim, {hid})
+         self.fc2 = nn.Linear({hid}, 1)
+
+     def forward(self, x):
+         x = torch.relu(self.fc1(x))
+         return self.fc2(x)  # raw logits, no sigmoid
+
+ model = BinaryClassifier({in_val}).cuda()
+ criterion = nn.BCEWithLogitsLoss()
+ optimizer = optim.Adam(model.parameters(), lr={lr})
```""",
    # --- Correct CrossEntropyLoss usage (no softmax, long targets) ---
    """\
```diff
+ model = nn.Sequential(
+     nn.Linear({in_val}, {hid}),
+     nn.ReLU(),
+     nn.Linear({hid}, {n_classes})
+ ).cuda()
+
+ criterion = nn.CrossEntropyLoss()
+ optimizer = optim.{optimizer}(model.parameters(), lr={lr})
+
+ for batch_x, batch_y in train_loader:
+     optimizer.zero_grad()
+     logits = model(batch_x.cuda())
+     loss = criterion(logits, batch_y.long().cuda())
+     loss.backward()
+     optimizer.step()
```""",
    # --- Correct training loop with .item() and shuffle ---
    """\
```diff
+ train_loader = DataLoader(dataset, batch_size={bs}, shuffle=True, num_workers=4)
+ losses = []
+
+ for epoch in range({epochs}):
+     model.train()
+     for batch_x, batch_y in train_loader:
+         optimizer.zero_grad()
+         out = model(batch_x.cuda())
+         loss = criterion(out, batch_y.cuda())
+         loss.backward()
+         optimizer.step()
+         losses.append(loss.item())
```""",
    # --- Correct eval mode + no_grad ---
    """\
```diff
+ model.eval()
+ with torch.no_grad():
+     total_correct = 0
+     for batch_x, batch_y in val_loader:
+         preds = model(batch_x.cuda())
+         total_correct += (preds.argmax(1) == batch_y.cuda()).sum().item()
+ accuracy = total_correct / len(val_loader.dataset)
+ model.train()
```""",
    # --- Correct preprocessing with fit on train only ---
    """\
```diff
+ from sklearn.preprocessing import {scaler}
+ from sklearn.model_selection import train_test_split
+
+ X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
+ scaler = {scaler}()
+ X_train_scaled = scaler.fit_transform(X_train)
+ X_test_scaled = scaler.transform(X_test)
+ model.fit(X_train_scaled, y_train)
```""",
    # --- Correct time-series split ---
    """\
```diff
+ from sklearn.model_selection import TimeSeriesSplit
+
+ tscv = TimeSeriesSplit(n_splits=5)
+ for train_idx, val_idx in tscv.split(X):
+     X_train, X_val = X[train_idx], X[val_idx]
+     y_train, y_val = y[train_idx], y[val_idx]
+     model.fit(X_train, y_train)
+     score = model.score(X_val, y_val)
```""",
    # --- Correct SMOTE inside pipeline ---
    """\
```diff
+ from imblearn.pipeline import make_pipeline
+ from imblearn.over_sampling import SMOTE
+
+ pipe = make_pipeline(SMOTE(random_state=42), RandomForestClassifier(n_estimators=100))
+ scores = cross_val_score(pipe, X, y, cv=5, scoring='f1')
+ print(f"CV F1: {{scores.mean():.4f}}")
```""",
    # --- Correct imbalanced data evaluation ---
    """\
```diff
+ from sklearn.metrics import classification_report, f1_score
+
+ predictions = model.predict(X_test)
+ print(classification_report(y_test, predictions))
+ f1 = f1_score(y_test, predictions, average='macro')
+ print(f"Macro F1: {{f1:.4f}}")
```""",
]

CAT21_REVIEWS = [
    "No critical ML bugs detected. The code correctly uses raw logits with `BCEWithLogitsLoss` (no double sigmoid), targets are properly typed, and the training loop follows best practices.",
    "This code looks correct. The loss function matches the model output format, `optimizer.zero_grad()` is called before backward, and `loss.item()` is used for logging. No action required.",
    "No issues found. The preprocessing pipeline correctly fits the scaler on training data only and applies `transform` to the test set. No data leakage.",
    "Clean implementation. The model is properly switched to `.eval()` mode with `torch.no_grad()` for validation, and back to `.train()` for the next epoch. No bugs.",
    "No ML anti-patterns detected. The DataLoader uses `shuffle=True` for training, the loss function is compatible with the model output, and gradient management is correct.",
    "This code follows ML best practices. SMOTE is correctly applied inside the cross-validation pipeline via `make_pipeline`, preventing data leakage. No changes needed.",
    "Correct evaluation approach. Using `classification_report` and `f1_score` instead of raw accuracy is the right choice for imbalanced datasets. No bugs found.",
    "No critical issues. The time-series data is split chronologically using `TimeSeriesSplit`, preserving temporal ordering. The evaluation is unbiased.",
]

CAT21_PARAMS = [
    {"hid": "128", "in_val": "50", "lr": "0.001", "n_classes": "10", "optimizer": "Adam", "bs": "64", "epochs": "50", "scaler": "StandardScaler"},
    {"hid": "256", "in_val": "768", "lr": "0.0001", "n_classes": "100", "optimizer": "AdamW", "bs": "32", "epochs": "30", "scaler": "MinMaxScaler"},
    {"hid": "512", "in_val": "100", "lr": "0.01", "n_classes": "5", "optimizer": "SGD", "bs": "128", "epochs": "100", "scaler": "RobustScaler"},
    {"hid": "64", "in_val": "384", "lr": "0.0005", "n_classes": "20", "optimizer": "Adam", "bs": "16", "epochs": "200", "scaler": "StandardScaler"},
]


CATEGORIES = [
    ("Data Leakage (fit_transform)", CAT1_CODE, CAT1_REVIEWS, CAT1_PARAMS, 60),
    ("GPU Memory Leak (loss.item)", CAT2_CODE, CAT2_REVIEWS, CAT2_PARAMS, 60),
    ("Missing model.eval()", CAT3_CODE, CAT3_REVIEWS, CAT3_PARAMS, 55),
    ("Missing torch.no_grad()", CAT4_CODE, CAT4_REVIEWS, CAT4_PARAMS, 50),
    ("Double Sigmoid", CAT5_CODE, CAT5_REVIEWS, CAT5_PARAMS, 60),
    ("Double Softmax", CAT6_CODE, CAT6_REVIEWS, CAT6_PARAMS, 50),
    ("Broadcasting Shape Mismatch", CAT7_CODE, CAT7_REVIEWS, CAT7_PARAMS, 55),
    ("Sequential DataLoader", CAT8_CODE, CAT8_REVIEWS, CAT8_PARAMS, 50),
    ("SMOTE before CV", CAT9_CODE, CAT9_REVIEWS, CAT9_PARAMS, 50),
    ("Time-Series Random Split", CAT10_CODE, CAT10_REVIEWS, CAT10_PARAMS, 50),
    ("Accuracy on Imbalanced Data", CAT11_CODE, CAT11_REVIEWS, CAT11_PARAMS, 50),
    ("Augmentation on Val Set", CAT12_CODE, CAT12_REVIEWS, CAT12_PARAMS, 45),
    ("CrossEntropyLoss Wrong Type", CAT13_CODE, CAT13_REVIEWS, CAT13_PARAMS, 50),
    (".view() on Permuted Tensor", CAT14_CODE, CAT14_REVIEWS, CAT14_PARAMS, 45),
    ("Missing zero_grad()", CAT15_CODE, CAT15_REVIEWS, CAT15_PARAMS, 50),
    ("In-place Ops Break Autograd", CAT16_CODE, CAT16_REVIEWS, CAT16_PARAMS, 40),
    ("RNN Hidden State Not Detached", CAT17_CODE, CAT17_REVIEWS, CAT17_PARAMS, 45),
    ("Target Leakage via Features", CAT18_CODE, CAT18_REVIEWS, CAT18_PARAMS, 50),
    ("No Scaling for Distance Models", CAT19_CODE, CAT19_REVIEWS, CAT19_PARAMS, 50),
    ("Tokenizer Fitted on Full Data", CAT20_CODE, CAT20_REVIEWS, CAT20_PARAMS, 45),
    ("Clean Code (No Bugs)", CAT21_CODE, CAT21_REVIEWS, CAT21_PARAMS, 100),
]


def generate_all():
    """Generate all SFT examples from all categories."""
    all_examples = []

    for name, codes, reviews, params, n_target in CATEGORIES:
        examples = _generate_category(codes, reviews, params, n_target)
        print(f"  • {name:40s}: {len(examples):>4} examples")
        all_examples.extend(examples)

    random.shuffle(all_examples)
    return all_examples


def save_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  ✓ Wrote {len(records):>4} examples → {path}")


def main():
    print("Generating diverse SFT dataset...\n")
    examples = generate_all()

    print(f"\n{'='*50}")
    print(f"Total examples generated: {len(examples)}")
    print(f"{'='*50}\n")

    # 90/10 split (larger training set for the bigger dataset)
    split_idx = int(len(examples) * 0.9)
    train_set = examples[:split_idx]
    valid_set = examples[split_idx:]

    save_jsonl(train_set, TRAIN_PATH)
    save_jsonl(valid_set, VALID_PATH)
    print("\nDone ✓")


if __name__ == "__main__":
    main()
