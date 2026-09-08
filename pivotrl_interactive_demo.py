# Databricks notebook source
# MAGIC %md
# MAGIC # 🍽️ The Gourmet Chef Soufflé Challenge
# MAGIC ## Interactive demo — why **PivotRL** beats End-to-End GRPO
# MAGIC
# MAGIC A 3-turn agentic task, trained two ways:
# MAGIC
# MAGIC | Turn | State | Under GRPO |
# MAGIC |---|---|---|
# MAGIC | 1️⃣ **Preparation** | Grease & sugar the ramekins | *Everything works* → reward variance **0** → **zero gradient** |
# MAGIC | 2️⃣ **Baking** 🔥 | Top browning too fast — rescue it! | *Mixed outcomes* → **high variance** → **the pivot** |
# MAGIC | 3️⃣ **Plating** | Add a sweet topping | *Any sweet topping works* → variance **0** → **zero gradient** |
# MAGIC
# MAGIC **The idea (PivotRL, [arXiv:2603.21383](https://arxiv.org/abs/2603.21383)):** don't waste
# MAGIC rollout compute on turns that teach the model nothing. Run cheap offline rollouts,
# MAGIC keep only the **high-variance pivots**, and train there. Result in the paper:
# MAGIC **+10.04% OOD accuracy** at **4× fewer rollout turns**.
# MAGIC
# MAGIC > Cells 1–3 are pure Python (instant, no GPU). Cell 4 runs the real GRPO training on the T4.

# COMMAND ----------

# MAGIC %pip install --quiet torch transformers accelerate pandas tabulate matplotlib

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

from kitchen_env import GourmetChefEnv, TURNS, TURN_BAKE, CANDIDATE_ACTIONS

env = GourmetChefEnv()
print("Loaded env from", _here)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎛️ Controls
# MAGIC Set the widgets at the top of the notebook, then **re-run** the cell you want to update.
# MAGIC - **turn2_action** — which action to try on the dangerous Baking turn
# MAGIC - **G**, **K**, **beta** — GRPO group size, offline dry-rollouts, KL coefficient

# COMMAND ----------

# Create the interactive widgets (dropdowns / text).
dbutils.widgets.removeAll()
dbutils.widgets.dropdown(
    "turn2_action",
    CANDIDATE_ACTIONS[TURN_BAKE][0],  # default: the correct rescue (index 0)
    CANDIDATE_ACTIONS[TURN_BAKE],
    "🔥 Turn 2 action",
)
dbutils.widgets.dropdown("G", "4", ["2", "4", "6", "8"], "GRPO group size G")
dbutils.widgets.dropdown("K", "4", ["2", "4", "6", "8"], "Offline dry-rollouts K")
dbutils.widgets.dropdown("beta", "0.02", ["0.0", "0.01", "0.02", "0.05", "0.1"], "KL coeff beta")
print("Widgets ready — see the control bar at the top of the notebook.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1️⃣ The Pivot Filter, live
# MAGIC Reward **variance** per turn over its rollout group. Flat bars (variance 0) are
# MAGIC **discarded** — GRPO can learn nothing from them. Only the spike is a **pivot**.

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({"font.size": 12, "axes.grid": True, "axes.axisbelow": True})

profiles = env.profile_all()
turns = list(TURNS)
variances = [profiles[t]["variance"] for t in turns]
means = [profiles[t]["mean"] for t in turns]
is_pivot = [profiles[t]["is_pivot"] for t in turns]

labels = ["1️⃣ Prep", "2️⃣ Baking", "3️⃣ Plating"]
colors = ["#c9ccd1" if not p else "#ff5f1f" for p in is_pivot]

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

bars = ax[0].bar(labels, variances, color=colors, edgecolor="#333", linewidth=1.2)
ax[0].set_title("Reward VARIANCE per turn  σ̂²(s)", fontweight="bold")
ax[0].set_ylabel("variance across rollout group")
ax[0].set_ylim(0, max(variances) * 1.35 + 0.05)
for b, t in zip(bars, turns):
    tag = "◀ PIVOT — TRAIN" if profiles[t]["is_pivot"] else "discard (0 grad)"
    ax[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, tag,
               ha="center", va="bottom", fontweight="bold",
               color="#ff5f1f" if profiles[t]["is_pivot"] else "#888")

ax[1].bar(labels, means, color=colors, edgecolor="#333", linewidth=1.2)
ax[1].axhline(env.lambda_diff, ls="--", color="#0072B2", lw=1.5,
              label=f"λ_diff = {env.lambda_diff}")
ax[1].set_title("Reward MEAN per turn  μ̂(s)", fontweight="bold")
ax[1].set_ylabel("mean reward")
ax[1].set_ylim(0, 1.15)
ax[1].legend(loc="lower center")

fig.suptitle("Offline Pivot Filtering:  keep turns with  σ̂²(s) > 0  AND  μ̂(s) < λ_diff",
             fontweight="bold", fontsize=13)
plt.tight_layout()
display(fig)

kept = [t for t in TURNS if profiles[t]["is_pivot"]]
print(f"Pivot Filter keeps: {kept}   |   discards: {[t for t in TURNS if t not in kept]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2️⃣ Try to save the soufflé  🥄
# MAGIC Pick **`turn2_action`** in the widget bar and re-run this cell. Only backing off
# MAGIC the heat (tent + lower temp) saves it — this is *why* Turn 2 has high variance.

# COMMAND ----------

action = dbutils.widgets.get("turn2_action")
reward = env.get_functional_reward(TURN_BAKE, action)

if reward >= 1.0:
    emoji, verdict, bg, border = "✨🎉", "RISES PERFECTLY — reward 1.0", "#e8f6ec", "#2e8b57"
elif "open" in action.lower():
    emoji, verdict, bg, border = "💥", "COLLAPSES — reward 0.0", "#fbe9e7", "#c62828"
else:
    emoji, verdict, bg, border = "🔥", "BURNS — reward 0.0", "#fbe9e7", "#c62828"

# Show the whole group's outcomes so the variance is visceral.
rows = ""
for a in CANDIDATE_ACTIONS[TURN_BAKE]:
    r = env.get_functional_reward(TURN_BAKE, a)
    ok = "✅ 1.0" if r >= 1.0 else "❌ 0.0"
    hl = "font-weight:700;background:#fff3cd;" if a == action else ""
    rows += f"<tr style='{hl}'><td style='padding:6px 12px'>{a}</td><td style='padding:6px 12px;text-align:center'>{ok}</td></tr>"

displayHTML(f"""
<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif">
  <div style="background:{bg};border:2px solid {border};border-radius:14px;padding:22px;margin-bottom:14px">
    <div style="font-size:52px;line-height:1">{emoji}</div>
    <div style="font-size:22px;font-weight:800;color:{border};margin-top:6px">{verdict}</div>
    <div style="color:#555;margin-top:4px">Your action: <b>{action}</b></div>
  </div>
  <table style="border-collapse:collapse;width:100%;max-width:640px;border:1px solid #ddd">
    <tr style="background:#222;color:#fff"><th style="padding:8px 12px;text-align:left">Turn-2 rollout (the group)</th><th style="padding:8px 12px">reward</th></tr>
    {rows}
  </table>
  <p style="color:#666;margin-top:10px">Mixed 0.0 / 1.0 outcomes ⇒ <b>high variance</b> ⇒ this turn is the <b>pivot</b> GRPO should train on.</p>
</div>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3️⃣ Why zero variance = zero learning (the GRPO math)
# MAGIC GRPO normalizes each action's reward against its group: **Aᵢ = (rᵢ − μ) / (σ + ε)**.
# MAGIC When every reward is identical, **σ = 0** and every advantage collapses to **0** — no gradient.

# COMMAND ----------

import statistics

def advantage_table(turn, title, note):
    actions = env.candidate_actions(turn)
    rewards = [env.get_functional_reward(turn, a) for a in actions]
    mean = statistics.fmean(rewards)
    std = statistics.pvariance(rewards) ** 0.5
    rows = ""
    for a, r in zip(actions, rewards):
        adv = 0.0 if std < 1e-9 else (r - mean) / (std + 1e-8)
        color = "#2e8b57" if adv > 0.01 else ("#c62828" if adv < -0.01 else "#999")
        rows += (f"<tr><td style='padding:5px 12px'>{a}</td>"
                 f"<td style='padding:5px 12px;text-align:center'>{r:.1f}</td>"
                 f"<td style='padding:5px 12px;text-align:center;color:{color};font-weight:700'>{adv:+.3f}</td></tr>")
    grad = "🚫 <b>zero gradient</b> — wasted rollouts" if std < 1e-9 else "✅ <b>real gradient signal</b>"
    return f"""
    <div style="flex:1;min-width:340px">
      <h3 style="margin:0 0 4px">{title}</h3>
      <div style="color:#666;font-size:13px;margin-bottom:6px">{note}</div>
      <table style="border-collapse:collapse;width:100%;border:1px solid #ddd;font-size:13px">
        <tr style="background:#222;color:#fff"><th style="padding:6px 12px;text-align:left">action</th><th style="padding:6px">rᵢ</th><th style="padding:6px">Aᵢ</th></tr>
        {rows}
      </table>
      <div style="margin-top:6px">μ={mean:.3f}, σ={std:.3f} → {grad}</div>
    </div>"""

displayHTML(f"""
<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;display:flex;gap:24px;flex-wrap:wrap">
  {advantage_table('preparation', '1️⃣ Preparation (flat)', 'All actions succeed — σ = 0.')}
  {advantage_table(TURN_BAKE, '2️⃣ Baking (the pivot)', 'Mixed outcomes — σ &gt; 0.')}
</div>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4️⃣ The money shot — run both trainers on the T4 ⚙️
# MAGIC Trains a real 0.5B policy under **End-to-End GRPO** (all turns) vs **PivotRL**
# MAGIC (pivot only), then compares rollout-turns, wall-clock, and OOD drift.

# COMMAND ----------

import train_comparative_grpo as tc

G = dbutils.widgets.get("G")
K = dbutils.widgets.get("K")
beta = dbutils.widgets.get("beta")

res_s, res_a, res_b = tc.main([
    "--model", "Qwen/Qwen2.5-0.5B-Instruct",
    "--epochs", "3",
    "--group-size", G,
    "--k", K,
    "--beta", beta,
    "--device", "auto",
    "--mode", "all",
])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🏆 Comparative scoreboard

# COMMAND ----------

turn_speedup = res_a.rollout_turns_train / max(res_b.rollout_turns_train, 1)
retained = max(0.0, (res_s.ood_kl - res_b.ood_kl) / max(res_s.ood_kl, 1e-9))
proj_ood = retained * tc.PAPER_OOD_CEILING_PCT


def metric_rows():
    data = [
        ("Turns trained on", "all 3 (imitate)", "all 3", ", ".join(res_b.trained_turns)),
        ("Online rollout-turns", res_s.rollout_turns_train, res_a.rollout_turns_train, res_b.rollout_turns_train),
        ("Total rollout-turns (+offline)",
         res_s.rollout_turns_offline + res_s.rollout_turns_train,
         res_a.rollout_turns_offline + res_a.rollout_turns_train,
         res_b.rollout_turns_offline + res_b.rollout_turns_train),
        ("Wall-clock (s)", f"{res_s.wall_clock_s:.2f}", f"{res_a.wall_clock_s:.2f}", f"{res_b.wall_clock_s:.2f}"),
        ("OOD drift  KL(π_θ‖π₀)", f"{res_s.ood_kl:.5f}", f"{res_a.ood_kl:.5f}", f"{res_b.ood_kl:.5f}"),
        ("P(rescue) at pivot", f"{res_s.pivot_confidence:.3f}", f"{res_a.pivot_confidence:.3f}", f"{res_b.pivot_confidence:.3f}"),
    ]
    out = ""
    for row in data:
        name, vs, va, vb = row
        out += (f"<tr><td style='padding:8px 14px;font-weight:600'>{name}</td>"
                f"<td style='padding:8px 14px;text-align:center'>{vs}</td>"
                f"<td style='padding:8px 14px;text-align:center'>{va}</td>"
                f"<td style='padding:8px 14px;text-align:center;background:#fff3cd;font-weight:700'>{vb}</td></tr>")
    return out

displayHTML(f"""
<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px">
  <table style="border-collapse:collapse;width:100%;border:1px solid #ccc">
    <tr style="background:#111;color:#fff">
      <th style="padding:10px 14px;text-align:left">Metric</th>
      <th style="padding:10px 14px">SFT (imitation)</th>
      <th style="padding:10px 14px">E2E GRPO</th>
      <th style="padding:10px 14px">PivotRL</th>
    </tr>
    {metric_rows()}
  </table>
  <div style="display:flex;gap:16px;margin-top:18px;flex-wrap:wrap">
    <div style="flex:1;min-width:170px;background:#ff5f1f;color:#fff;border-radius:14px;padding:18px;text-align:center">
      <div style="font-size:34px;font-weight:800">{turn_speedup:.1f}×</div>
      <div>fewer online rollout-turns<br>than E2E GRPO</div>
    </div>
    <div style="flex:1;min-width:170px;background:#2e8b57;color:#fff;border-radius:14px;padding:18px;text-align:center">
      <div style="font-size:34px;font-weight:800">{retained*100:.0f}%</div>
      <div>of SFT's lost OOD<br>capability retained</div>
    </div>
    <div style="flex:1;min-width:170px;background:#0072B2;color:#fff;border-radius:14px;padding:18px;text-align:center">
      <div style="font-size:34px;font-weight:800">+{proj_ood:.2f}%</div>
      <div>projected OOD gain vs SFT<br>(ceiling +{tc.PAPER_OOD_CEILING_PCT:.2f}%)</div>
    </div>
  </div>
  <p style="color:#666;margin-top:14px;font-size:13px">
    <b>SFT</b> is cheapest (0 online rollouts) but drifts furthest from π₀ (worst OOD).
    <b>E2E GRPO</b> retains OOD but pays for rolling out every turn — including the
    zero-gradient prep &amp; plating turns. <b>PivotRL</b> trains only the high-variance
    baking pivot: E2E-level OOD retention at a fraction of the rollout compute. The
    projected OOD gain is an illustrative scaling toward the paper's reported
    +{tc.PAPER_OOD_CEILING_PCT:.2f}%.
  </p>
</div>
""")
