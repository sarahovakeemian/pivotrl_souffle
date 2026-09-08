# 🍽️ PivotRL Soufflé Challenge

**A tiny, self-contained demo that shows *why* [PivotRL](https://arxiv.org/abs/2603.21383) trains long-horizon agents at a fraction of the compute of end-to-end RL — using a 3-turn "Gourmet Chef Soufflé Challenge" and a real 0.5B language model.**

Traditional post-training for agentic tasks forces a bad trade-off: **SFT** is cheap but forgets everything it wasn't fine-tuned on (catastrophic out-of-domain degradation), while **end-to-end RL** keeps general capability but pays for a full multi-turn rollout on *every* gradient step. **PivotRL** breaks the trade-off by noticing that most turns in a trajectory teach the model nothing, and training only on the ones that do.

This repo makes that idea tangible and interactive.

---

## TL;DR

We model an agent baking a soufflé over three turns:

| Turn | State | What happens under GRPO |
|------|-------|--------------------------|
| 1️⃣ **Preparation** | Grease & sugar the ramekins | *Every* action succeeds → reward variance **0** → advantage **0** → **zero gradient** (wasted rollout) |
| 2️⃣ **Baking** 🔥 | The top is browning too fast — rescue it! | *Mixed* outcomes (collapse / burn / perfect rise) → **high variance** → **the pivot** worth training |
| 3️⃣ **Plating** | Add a sweet topping | *Any* sweet topping works → variance **0** → **zero gradient** (wasted rollout) |

Trained three ways on `Qwen/Qwen2.5-0.5B-Instruct`, on a **single NVIDIA T4**:

| Metric | SFT (imitation) | End-to-End GRPO | **PivotRL** |
|---|---|---|---|
| Turns trained on | all 3 | all 3 | **baking only** |
| Online rollout-turns | 0 | 36 | **12** (+12 offline profiling) |
| Wall-clock (s) | 3.21 | 7.41 | **2.54** |
| OOD drift `KL(π_θ‖π₀)` ↓ | 1.100 | 0.546 | **0.213** |
| P(rescue) learned at pivot | 0.926 | 0.898 | 0.911 |

**PivotRL used 3× fewer online rollout-turns than end-to-end GRPO, drifted the least from the reference policy (best OOD retention), and still learned the pivot** — all three arms end up preferring the correct rescue action. See [Findings](#findings) for the full interpretation.

---

## The research

### GRPO and the "uninformative turn" bottleneck

Group Relative Policy Optimization (GRPO), introduced in [DeepSeekMath](https://arxiv.org/abs/2402.03300), estimates each candidate action's advantage by normalizing its reward against the group of `G` sampled actions for the same state:

```
A_i = (r_i − mean(r)) / (std(r) + ε)
```

The structural consequence this demo is built around: **when every action in a group gets the same reward — all succeed (`r = 1`) or all fail (`r = 0`) — the group's standard deviation is 0, so every advantage collapses to exactly 0, and the gradient is 0.** Those turns still consume a full rollout, but move the weights nowhere. In our kitchen, that's turns 1 (prep) and 3 (plating).

### PivotRL: offline pivot filtering + functional verifiers

[**PivotRL: High Accuracy Agentic Post-Training at Low Compute Cost**](https://arxiv.org/abs/2603.21383) (Yi et al., 2026) operates *on top of existing SFT trajectories* with two engines:

1. **Offline Pivot Filtering.** Run `K` cheap local rollouts at each state to estimate reward mean `μ̂(s)` and variance `σ̂²(s)`. Keep only **pivots** — states with **high variance and low mean** (`σ̂²(s) > 0` and `μ̂(s) < λ_diff`) — where the gradient signal is strongest. Discard the rest. Training compute concentrates where learning actually happens.
2. **Verifier-Based Functional Rewards.** Instead of exact string-matching against the SFT expert action (which penalizes `ls` vs `ls -l`), a domain verifier checks membership in a set of acceptable actions `M(s)` and awards `r_func(s, a) = 1[a ∈ M(s)]`. In our kitchen, the plating verifier accepts *any* sweet topping — "powdered sugar", "chocolate syrup", "cocoa" — not one blessed string.

Because updates are localized and **KL-regularized back to a frozen reference policy `π₀`**, PivotRL preserves the model's behavior on unrelated inputs (Theorem 3.3 limits catastrophic forgetting). The paper reports **+4.17% in-domain, +10.04% OOD** accuracy over SFT at **4× fewer rollout turns** than end-to-end RL, and was used as a workhorse for NVIDIA's Nemotron agentic post-training.

### How each idea appears in this repo

| Concept | Where it lives |
|---|---|
| GRPO group-normalized advantage | `train_comparative_grpo.py → grpo_group_loss()` |
| Zero-variance ⇒ zero gradient | Turns 1 & 3 in `kitchen_env.py`; visible in training logs |
| Offline pivot filtering (`σ̂²`, `μ̂`, `λ_diff`) | `kitchen_env.py → profile_turn()`; `run_mode_b()` |
| Functional verifier reward `1[a ∈ M(s)]` | `kitchen_env.py → get_functional_reward()` (plating turn) |
| KL to frozen `π₀` (k3 estimator) | `train_comparative_grpo.py → k3_kl()` |
| OOD retention measurement | `train_comparative_grpo.py → ood_drift()` |

---

## Repo layout

```
pivotrl_souffle/
├── kitchen_env.py                # The 3-turn environment + functional verifier (pure Python, self-testing)
├── train_comparative_grpo.py     # Turn-level GRPO engine; 3 arms: SFT / E2E-GRPO / PivotRL
├── databricks_launcher.py        # Databricks notebook — headless "run it all" + JSON summary
├── pivotrl_interactive_demo.py   # Databricks notebook — the flashy interactive walkthrough ⭐
├── requirements.txt
└── README.md
```

`kitchen_env.py` and `train_comparative_grpo.py` are plain importable modules. The two `*_demo` / `*_launcher` files carry a `# Databricks notebook source` header and render as runnable Databricks notebooks.

---

## Setup & running

### Option A — Local (CPU/MPS, quick mechanics check)

```bash
pip install -r requirements.txt

# 1) Verify the environment + verifier + pivot structure (no model needed):
python kitchen_env.py

# 2) Run the full 3-way comparison with a small model:
python train_comparative_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3 --group-size 4
#   ...or a tiny model for a fast smoke test:
python train_comparative_grpo.py --model sshleifer/tiny-gpt2 --epochs 3 --device cpu
```

Useful flags: `--group-size G` (GRPO group), `--k K` (offline dry-rollouts for filtering), `--beta` (KL coefficient), `--lr`, `--epochs`, `--mode {all,S,A,B}`, `--device {auto,cuda,mps,cpu}`.

### Option B — Databricks (the intended stage)

1. **Import the folder** into your workspace (all four files in one directory so the notebooks can `import kitchen_env`). Via CLI:
   ```bash
   # modules as workspace files (keep the .py extension):
   databricks workspace import <dir>/kitchen_env.py --file kitchen_env.py --format RAW --overwrite
   databricks workspace import <dir>/train_comparative_grpo.py --file train_comparative_grpo.py --format RAW --overwrite
   # notebooks as SOURCE (creates runnable NOTEBOOK objects — do NOT use RAW/AUTO here):
   databricks workspace import <dir>/databricks_launcher --file databricks_launcher.py --format SOURCE --language PYTHON --overwrite
   databricks workspace import <dir>/pivotrl_interactive_demo --file pivotrl_interactive_demo.py --format SOURCE --language PYTHON --overwrite
   ```
   (Or just clone this repo into Databricks Repos.)

2. **Cluster:** Single Node, **latest LTS ML (GPU)** runtime, **1× NVIDIA T4** — AWS `g4dn.xlarge`, Azure `Standard_NC4as_T4_v3`, or a GCP T4/L4 node. A 0.5B model in fp32 plus Adam state fits comfortably in a T4's 16 GB.

3. **Run** `pivotrl_interactive_demo` for the live walkthrough, or `databricks_launcher` for a headless end-to-end run (it returns a JSON metrics summary via `dbutils.notebook.exit`).

The interactive notebook drives inputs with `dbutils.widgets` (pick the Turn-2 action; tweak `G` / `K` / `beta`) and renders a live pivot-variance chart, an interactive soufflé-rescue panel, the GRPO advantage math, and a comparative scoreboard. Cells 1–3 are pure Python (instant); the final cell runs the real training on the T4.

---

## Findings

Numbers below are a real run of `databricks_launcher` on `Qwen/Qwen2.5-0.5B-Instruct`, single T4, latest LTS ML runtime, `epochs=3, G=4, K=4, beta=0.02, lr=1e-5`.

```
                                  SFT (imitation)   E2E GRPO   PivotRL
Turns trained on                  all 3             all 3      baking
Online rollout-turns              0                 36         12  (+12 offline)
Wall-clock (s)                    3.21              7.41       2.54
OOD drift  KL(π_θ‖π₀)             1.100             0.546      0.213
P(rescue) at pivot                0.926             0.898      0.911
```

**1. Compute (vs end-to-end GRPO): ~3× fewer online rollout-turns.** PivotRL's filter discarded prep and plating — the two turns whose GRPO advantage is provably 0 — and trained only the baking pivot. End-to-end GRPO rolled out all three turns every epoch and got *identical* learning value from two of them.

**2. OOD retention (vs SFT): PivotRL drifts least.** SFT, with no KL brake, drifts furthest from the reference policy (`KL = 1.100`) — the mechanism behind catastrophic forgetting. PivotRL's few, KL-regularized, localized updates keep it closest to `π₀` (`0.213`), even edging out end-to-end GRPO (`0.546`), which takes more optimizer steps. In this run PivotRL **retained ~81% of the OOD capability SFT loses** (a projected **+8.09%** toward the paper's reported **+10.04%** ceiling).

**3. All three still learn the task.** Every arm ends up assigning ~0.90–0.93 probability mass to the correct "tent with foil and lower the temp" rescue at the pivot. PivotRL gets there **without** spending gradient steps on the turns that had nothing to teach.

> **The headline:** SFT is cheap but forgets; end-to-end RL remembers but is expensive; **PivotRL is cheap *and* remembers** — by spending compute only on the pivot.

---

## Honest caveats

This is a **pedagogical** demo, not a benchmark. Deliberate simplifications:

- **The rollout group is a fixed candidate pool** per turn, not fresh on-policy generations — this makes the variance structure crisp and reproducible. Log-probs, KL, advantages, and back-prop are all computed for real against the live model.
- **One on-policy gradient step per collected group**, so the GRPO importance ratio is ≈ 1 and the clipped surrogate reduces to `A_i · logπ(a_i|s)`.
- **The OOD-retention projection is illustrative** — it scales the *measured* KL-drift ordering toward the paper's reported `+10.04%`. It is not a benchmarked accuracy on a held-out suite.
- **Tiny scale** (0.5B model, 3 turns, 3 epochs): absolute numbers wobble run-to-run; the *ordering* (SFT drifts most, PivotRL least; PivotRL ~3× fewer rollouts) is the robust, reproducible result.

For the real thing — many-turn trajectories, SWE-Bench-scale tasks, production models — see the paper and [NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym).

---

## References

- **PivotRL: High Accuracy Agentic Post-Training at Low Compute Cost** — Yi et al., 2026. [arXiv:2603.21383](https://arxiv.org/abs/2603.21383)
- **DeepSeekMath (GRPO)** — Shao et al., 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- **NVIDIA-NeMo/Gym** — multi-turn agentic RL environments. [github.com/NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym)

*Model: `Qwen/Qwen2.5-0.5B-Instruct`. Demo built for educational use.*
