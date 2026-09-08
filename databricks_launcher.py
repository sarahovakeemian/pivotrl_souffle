# Databricks notebook source
# MAGIC %md
# MAGIC # PivotRL Souffle Challenge — Headless Launcher
# MAGIC
# MAGIC Runs the full **comparative turn-level GRPO** demo end-to-end and prints the
# MAGIC side-by-side table: **Pre-PivotRL (End-to-End GRPO)** vs **PivotRL (Offline
# MAGIC Pivot Filtering)**.
# MAGIC
# MAGIC For the *interactive / flashy* version (live pivot-filter chart, soufflé
# MAGIC rescue widget, GRPO advantage explainer), open **`pivotrl_interactive_demo`**
# MAGIC in this same folder.
# MAGIC
# MAGIC ### Research background
# MAGIC - **GRPO** — DeepSeekMath, [arXiv:2402.03300](https://arxiv.org/abs/2402.03300):
# MAGIC   advantages are group-normalized `A_i = (r_i − mean)/(std+eps)`. A
# MAGIC   zero-variance group ⇒ every advantage 0 ⇒ **zero gradient** (wasted compute).
# MAGIC - **PivotRL** — Yi et al. 2026, [arXiv:2603.21383](https://arxiv.org/abs/2603.21383):
# MAGIC   offline pivot filtering keeps only high-variance / low-mean turns, plus a
# MAGIC   functional-verifier reward. Reported **+4.17% in-domain, +10.04% OOD** vs SFT
# MAGIC   at **4× fewer rollout turns**.
# MAGIC
# MAGIC ### Recommended cluster
# MAGIC | Setting | Value |
# MAGIC |---|---|
# MAGIC | Mode | **Single Node** |
# MAGIC | Runtime | **Latest LTS ML** (GPU), e.g. `*-cuda*-ml-gpu` |
# MAGIC | Node type (AWS) | **`g4dn.xlarge`** — 1× NVIDIA **T4** (16 GB) |
# MAGIC | Node type (Azure) | `Standard_NC4as_T4_v3` (1× T4) |
# MAGIC | Node type (GCP) | `g2-standard-4` (L4) or n1 + 1× T4 |
# MAGIC
# MAGIC A 0.5B model in fp32 plus Adam state fits comfortably in a single T4's 16 GB.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Dependencies
# MAGIC The Databricks **ML runtime already ships torch/transformers/accelerate/pandas**.
# MAGIC This `%pip` line pins the pieces the demo relies on and adds `tabulate` for the
# MAGIC ASCII comparison table. Safe to keep even on ML runtimes (it is a fast no-op /
# MAGIC light upgrade).

# COMMAND ----------

# MAGIC %pip install --quiet torch transformers accelerate pandas tabulate

# COMMAND ----------

# Restart the Python interpreter so freshly installed/updated wheels are picked up.
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Make the local modules importable
# MAGIC `kitchen_env.py` and `train_comparative_grpo.py` live in this same workspace
# MAGIC folder. Add the notebook's directory to `sys.path` so `import` finds them.

# COMMAND ----------

import os
import sys

# Resolve this notebook's folder and put it on the import path.
try:
    _nb_path = (
        dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    )
    _here = "/Workspace" + os.path.dirname(_nb_path)
except Exception:
    _here = os.getcwd()

if _here not in sys.path:
    sys.path.insert(0, _here)
print("Importing modules from:", _here)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Environment sanity check
# MAGIC Runs the self-test in `kitchen_env.py` — verifies the verifier rewards and the
# MAGIC pivot structure (baking = high variance; prep & plating = zero variance).

# COMMAND ----------

import torch
import kitchen_env
from kitchen_env import GourmetChefEnv

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

kitchen_env._selftest()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Run the comparative GRPO demo (both modes)
# MAGIC Trains a fresh copy of the policy under each regime and prints the comparison.
# MAGIC Small by design so it runs in well under a minute on a T4.

# COMMAND ----------

import train_comparative_grpo as tc

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

res_s, res_a, res_b = tc.main([
    "--model", MODEL,
    "--epochs", "3",
    "--group-size", "4",
    "--k", "4",
    "--beta", "0.02",
    "--lr", "1e-5",
    "--device", "auto",
    "--mode", "all",
])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Programmatic results
# MAGIC The two `ModeResult` objects are available for assertions / dashboards / MLflow.

# COMMAND ----------

import pandas as pd

summary = pd.DataFrame([
    {
        "mode": r.name,
        "trained_turns": ", ".join(r.trained_turns),
        "online_rollout_turns": r.rollout_turns_train,
        "offline_rollout_turns": r.rollout_turns_offline,
        "wall_clock_s": round(r.wall_clock_s, 2),
        "ood_kl_drift": round(r.ood_kl, 5),
        "p_rescue_at_pivot": round(r.pivot_confidence, 3),
    }
    for r in (res_s, res_a, res_b)
])
display(summary)

# COMMAND ----------

speedup = res_a.rollout_turns_train / max(res_b.rollout_turns_train, 1)
retained = max(0.0, (res_s.ood_kl - res_b.ood_kl) / max(res_s.ood_kl, 1e-9))
print(f"COMPUTE (vs E2E-GRPO): PivotRL used {speedup:.2f}x fewer online rollout-turns.")
print(f"OOD DRIFT KL(pi_theta||pi_0):  SFT={res_s.ood_kl:.5f}  "
      f"E2E={res_a.ood_kl:.5f}  PivotRL={res_b.ood_kl:.5f}")
print(f"PivotRL retained ~{retained*100:.0f}% of the OOD capability SFT loses "
      f"(projecting toward the paper's +{tc.PAPER_OOD_CEILING_PCT:.2f}%).")

# COMMAND ----------

# Return a compact JSON summary so automated runs can capture the results.
import json as _json


def _r(x):
    return {
        "name": x.name,
        "trained_turns": x.trained_turns,
        "online_rollout_turns": x.rollout_turns_train,
        "offline_rollout_turns": x.rollout_turns_offline,
        "wall_clock_s": round(x.wall_clock_s, 3),
        "ood_kl": round(x.ood_kl, 6),
        "pivot_confidence": round(x.pivot_confidence, 4),
    }


dbutils.notebook.exit(_json.dumps({
    "sft": _r(res_s),
    "e2e_grpo": _r(res_a),
    "pivotrl": _r(res_b),
    "rollout_turn_speedup_vs_e2e": round(speedup, 2),
    "ood_retained_vs_sft_pct": round(retained * 100, 1),
    "projected_ood_gain_pct": round(retained * tc.PAPER_OOD_CEILING_PCT, 2),
}))
