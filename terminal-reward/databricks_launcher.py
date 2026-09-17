# Databricks notebook source
# MAGIC %md
# MAGIC # PivotRL Soufflé — TERMINAL-REWARD Launcher
# MAGIC
# MAGIC The realistic variant: the environment gives a reward **only at the end**
# MAGIC (good soufflé = 1, ruined = 0), so PivotRL must **discover** which turn is the
# MAGIC pivot instead of being handed per-turn rewards.
# MAGIC
# MAGIC The headline of this run is the **discovery** step — watch it roll out to
# MAGIC completion and correctly isolate **baking** as the pivot, while noting that the
# MAGIC *naive* outcome-variance signal would have been fooled into also flagging prep.
# MAGIC
# MAGIC See `../dense-reward/` for the simpler teaching variant (per-turn rewards).
# MAGIC
# MAGIC ### Recommended cluster
# MAGIC Single Node · latest LTS ML (GPU) · 1× NVIDIA **T4** (AWS `g4dn.xlarge`).

# COMMAND ----------

# MAGIC %pip install --quiet torch transformers accelerate pandas tabulate

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

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
# MAGIC ### Environment sanity check (terminal reward only)

# COMMAND ----------

import kitchen_env
kitchen_env._selftest()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run the terminal-reward experiment
# MAGIC Base π₀ / SFT / End-to-End GRPO / PivotRL. PivotRL first **discovers** the pivot
# MAGIC from the terminal reward (`M` completion-rollouts per candidate), then trains it.

# COMMAND ----------

import train_terminal_grpo as tt

res_base, res_s, res_a, res_b = tt.main([
    "--model", "Qwen/Qwen2.5-0.5B-Instruct",
    "--epochs", "3",
    "--group-size", "4",
    "--m", "16",
    "--pivot-tau", "0.05",
    "--device", "auto",
    "--mode", "all",
])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Programmatic results + JSON summary

# COMMAND ----------

import pandas as pd

summary = pd.DataFrame([
    {
        "mode": r.name,
        "trained_turns": ", ".join(r.trained_turns) or "(none)",
        "discovery_rollouts": r.discovery_rollouts,
        "train_rollout_turns": r.train_rollout_turns,
        "wall_clock_s": round(r.wall_clock_s, 2),
        "ood_kl_drift": round(r.ood_kl, 5),
        "p_rescue_at_pivot": round(r.pivot_confidence, 3),
    }
    for r in (res_base, res_s, res_a, res_b)
])
display(summary)

# COMMAND ----------

import json as _json


def _r(x):
    return {
        "name": x.name,
        "trained_turns": x.trained_turns,
        "discovered_pivots": x.discovered_pivots,
        "discovery_rollouts": x.discovery_rollouts,
        "train_rollout_turns": x.train_rollout_turns,
        "wall_clock_s": round(x.wall_clock_s, 3),
        "ood_kl": round(x.ood_kl, 6),
        "pivot_confidence": round(x.pivot_confidence, 4),
    }


base_c = res_base.pivot_confidence
dbutils.notebook.exit(_json.dumps({
    "base": _r(res_base),
    "sft": _r(res_s),
    "e2e_grpo": _r(res_a),
    "pivotrl": _r(res_b),
    "base_pivot_confidence": round(base_c, 4),
    "discovered_pivots": res_b.discovered_pivots,
    "discovery_rollouts": res_b.discovery_rollouts,
    "train_rollout_turn_speedup_vs_e2e": round(res_a.train_rollout_turns / max(res_b.train_rollout_turns, 1), 2),
}))
