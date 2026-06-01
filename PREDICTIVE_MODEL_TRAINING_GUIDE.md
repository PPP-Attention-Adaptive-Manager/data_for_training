# AAM Predictive Model — Teammate Training Guide

> **Who this is for:** mouse, keyboard, notif, and switching model owners  
> **What you receive:** `tucker_slices.npy`, `nasa_tlx_labels.npy`, `metadata.json`  
> **What you deliver:** a trained model file + updated `__init__.py`

---

## 1. What you received

### `tucker_slices.npy`
Shape `(N, 4, 512)` — float32.

- `N` = total windows across all users and sessions
- axis 1 = modality index: `0=mouse, 1=keyboard, 2=notif, 3=switching`
- axis 2 = Tucker slice dimension (512 per modality)

Your model receives column `tucker_slices[i, YOUR_MODALITY_IDX, :]` — shape `(512,)` — as input `x` for window `i`.

**Your modality index:**

| Modality  | Index |
|-----------|-------|
| mouse     | 0     |
| keyboard  | 1     |
| notif     | 2     |
| switching | 3     |

### `nasa_tlx_labels.npy`
Shape `(N, 9)` — float32.

Column order (fixed, do not change):

| Index | Column              | Range  |
|-------|---------------------|--------|
| 0     | mental_demand       | 0–100  |
| 1     | physical_demand     | 0–100  |
| 2     | temporal_demand     | 0–100  |
| 3     | performance         | 0–100  |
| 4     | effort              | 0–100  |
| 5     | frustration         | 0–100  |
| 6     | stress_self_report  | 0–100  |
| 7     | valence             | 0–100  |
| 8     | arousal             | 0–100  |

**NASA-TLX is session-level**, not window-level. Every window in the same session has the same label row. This is correct and expected — subjects fill out TLX once per session.

### `metadata.json`
List of dicts, one per row:
```json
{
  "user_id":     "hffz",
  "session_id":  "session_20260520_220220_20835b",
  "window_idx":  0,
  "window_start": 1779310941.04,
  "window_end":   1779311061.04
}
```
Use this for LOSO splits — group by `user_id`.

---

## 2. Which labels to use for training

Your model outputs `(B, 12)`:
- dims `0–4` : factor scores → train with `nasa_tlx_labels[:, [0,2,4,5,8]]` (MD, TD, EF, FR, AR)
- dims `5–9` : state logits → derive from NASA-TLX using the function below

**The 5 factors your model predicts (dims 0–4):**

| Dim | Factor            | NASA-TLX column index |
|-----|-------------------|-----------------------|
| 0   | mental_demand     | 0                     |
| 1   | temporal_demand   | 2                     |
| 2   | effort            | 4                     |
| 3   | frustration       | 5                     |
| 4   | arousal (proxy)   | computed: (TD+EF)/2   |

**Deriving state labels from NASA-TLX:**

```python
import numpy as np

def derive_state_label(nasa_tlx_row: np.ndarray) -> int:
    """
    Derive cognitive state class from NASA-TLX scores.
    
    Returns:
        0 = Flow        (low demand, low frustration, high performance)
        1 = Neutral     (moderate everything)
        2 = Bored       (low demand, low frustration, low performance)
        3 = Distracted  (moderate demand + high temporal pressure)
        4 = Overloaded  (high demand + high frustration)
    """
    md   = float(nasa_tlx_row[0])   # mental demand
    td   = float(nasa_tlx_row[2])   # temporal demand
    ef   = float(nasa_tlx_row[4])   # effort
    fr   = float(nasa_tlx_row[5])   # frustration
    perf = float(nasa_tlx_row[3])   # performance (inverted: low = good)

    # primary discriminators
    high_demand = (md + ef) / 2 > 60
    high_frust  = fr > 60
    high_td     = td > 65
    low_demand  = (md + ef) / 2 < 35
    good_perf   = perf < 35   # NASA-TLX performance: low = performing well

    if high_demand and high_frust:
        return 4   # Overloaded
    if high_td and not high_frust:
        return 3   # Distracted
    if not high_demand and good_perf and not high_frust:
        return 0   # Flow
    if low_demand and not good_perf:
        return 2   # Bored
    return 1       # Neutral


def prepare_targets(labels: np.ndarray) -> tuple:
    """
    Prepare training targets from nasa_tlx_labels.npy.

    Args:
        labels: (N, 9) from nasa_tlx_labels.npy

    Returns:
        factors: (N, 5) float32 — [MD, TD, EF, FR, AR]
        states:  (N,)   int64   — class 0–4
    """
    md   = labels[:, 0:1]           # mental_demand
    td   = labels[:, 2:3]           # temporal_demand
    ef   = labels[:, 4:5]           # effort
    fr   = labels[:, 5:6]           # frustration
    ar   = (labels[:, 2:3] + labels[:, 4:5]) / 2   # arousal proxy

    factors = np.concatenate([md, td, ef, fr, ar], axis=1).astype(np.float32)
    states  = np.array(
        [derive_state_label(row) for row in labels], dtype=np.int64
    )
    return factors, states
```

---

## 3. Training loss

Use this exact loss function. Do not use softmax before returning logits.

```python
import torch
import torch.nn.functional as F

def compute_loss(output: torch.Tensor, factors_gt: torch.Tensor, states_gt: torch.Tensor) -> torch.Tensor:
    """
    Args:
        output:     (B, 12) — your model output
        factors_gt: (B, 5)  — NASA-TLX factor targets (normalized to 0-1)
        states_gt:  (B,)    — state class labels 0–4

    Returns:
        scalar loss
    """
    factor_loss = F.huber_loss(output[:, :5], factors_gt)
    state_loss  = F.cross_entropy(output[:, 5:10], states_gt)
    return 0.4 * factor_loss + 0.6 * state_loss
```

**Normalize factor targets before training:**
```python
factors_normalized = factors / 100.0   # scale [0,100] → [0,1]
```

---

## 4. LOSO split

Use `metadata.json` to build subject-based splits. Never shuffle windows across subjects.

```python
import json
import numpy as np

with open("metadata.json") as f:
    meta = json.load(f)

tucker = np.load("tucker_slices.npy")   # (N, 4, 512)
labels = np.load("nasa_tlx_labels.npy") # (N, 9)

# Get unique users
users = sorted(set(m["user_id"] for m in meta))

for test_user in users:
    train_idx = [i for i, m in enumerate(meta) if m["user_id"] != test_user]
    test_idx  = [i for i, m in enumerate(meta) if m["user_id"] == test_user]

    # YOUR_MODALITY_IDX: 0=mouse, 1=keyboard, 2=notif, 3=switching
    MODALITY_IDX = 0

    X_train = tucker[train_idx, MODALITY_IDX, :]   # (N_train, 512)
    X_test  = tucker[test_idx,  MODALITY_IDX, :]   # (N_test,  512)
    y_train = labels[train_idx]
    y_test  = labels[test_idx]

    # Normalize per-user (fit on train, transform both)
    mean = X_train.mean(axis=0, keepdims=True)
    std  = X_train.std(axis=0,  keepdims=True) + 1e-9
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    # train your model on (X_train, y_train)
    # evaluate on (X_test, y_test)
```

---

## 5. Model contract — what your model must satisfy

Subclass `BaseModalityModel` from `predictive_models/base.py`.

```python
from predictive_models.base import BaseModalityModel
from ema.ema import compute_uncertainty
import torch
import torch.nn as nn

class YourModalityModel(BaseModalityModel):

    def __init__(self, input_flat_dim: int, d_proj: int = 256, **kwargs):
        super().__init__(input_flat_dim=input_flat_dim, d_proj=d_proj, **kwargs)
        # build your architecture here
        # self.projector is already created by base: Linear(512, 256)
        self.gru = nn.GRU(d_proj, d_proj, batch_first=True)
        self.factor_head = nn.Linear(d_proj, 5)
        self.state_head  = nn.Linear(d_proj, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 512)
        feat = self.projector(x)                    # MANDATORY first line
        out, _ = self.gru(feat.unsqueeze(1))
        out = out.squeeze(1)
        factors = self.factor_head(out)             # (B, 5)
        logits  = self.state_head(out)              # (B, 5) — NO softmax
        H_norm, M = compute_uncertainty(logits)
        return torch.cat([
            factors,
            logits,
            H_norm.unsqueeze(-1),
            M.unsqueeze(-1),
        ], dim=-1)                                  # (B, 12)

    def reset_microstate(self):
        self.microstate = {}
```

**Hard rules:**
- First line of `forward()` must be `feat = self.projector(x)`
- dims `5–9` must be **raw logits** — never apply softmax before returning
- dims `10–11` must come from `compute_uncertainty(logits)`
- `reset_microstate()` must clear all hidden state

---

## 6. Plugging your model in

**Step 1** — Save your model file:
```
predictive_models/{your_modality}/v1_{your_name}.py
```

**Step 2** — Update the swap file (one line only):
```python
# predictive_models/{your_modality}/__init__.py
from .v1_your_name import YourModalityModel as ActiveModel
```

**Step 3** — Verify with the smoke test:
```bash
cd ~/fusion_model
python test_fusion.py
```
All 7 tests must pass.

---

## 7. Evaluation metrics to report

```python
from sklearn.metrics import f1_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix

# primary
macro_f1 = f1_score(y_true, y_pred, average="macro")
mcc      = matthews_corrcoef(y_true, y_pred)
kappa    = cohen_kappa_score(y_true, y_pred)

# per-class
cm = confusion_matrix(y_true, y_pred)
```

Report all four: macro F1, MCC, kappa, and the 5×5 confusion matrix.

---

## 8. Quick checklist before submitting

```
□ Model file saved to predictive_models/{modality}/v1_yourname.py
□ Subclasses BaseModalityModel
□ First line of forward() is self.projector(x)
□ Output shape is (B, 12)
□ dims 5–9 are raw logits (no softmax)
□ dims 10–11 computed via compute_uncertainty()
□ reset_microstate() clears hidden state
□ __init__.py updated to point at your model
□ test_fusion.py passes 7/7
□ LOSO results reported: macro F1, MCC, kappa, confusion matrix
```

---

*AAM Inférer — Fusion Model v3 | Contact: Haffouz (fusion lead)*
