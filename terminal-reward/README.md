# 🍽️ Terminal-Reward PivotRL — *discovering* the pivot

This is the **realistic** variant of the soufflé demo. See [`../dense-reward/`](../dense-reward) for the simpler teaching version.

## What's different

In the dense-reward demo, every turn hands out its own reward, so the pivot (the high-variance turn) is trivially visible — we basically *tell* the algorithm where to train. Real agentic tasks don't work that way. In code generation, math, or web search, you get **one reward, at the very end** — did the whole thing ultimately succeed? — and nothing in between. That's a **terminal reward**, and it forces the hard question PivotRL actually has to answer:

> With only an end-of-episode good/bad verdict, **how do you figure out which turn was the pivot?**

## How it discovers the pivot

For each turn, freeze the earlier turns to the expert (SFT) actions, try each candidate action there, let the **current policy finish the rest of the recipe**, and score the finished soufflé. Repeat `M` times per candidate. Then compare two signals:

| Signal | What it asks | Result |
|---|---|---|
| **Naive: pooled outcome variance** | "Is the final outcome uncertain from this state?" | **Fooled** — it lights up on *prep* too, because the downstream baking pivot bleeds into it |
| **Correct: action-value variance** | "Does *my choice at this turn* change the final outcome?" | **Isolates baking** — prep choices all lead to the same downstream lottery (variance ≈ 0), baking choices swing the ending from 0→1 |

Example discovery output (base Qwen2.5-0.5B):

```
turn         action-value var   pooled var   mean R   verdict
preparation            0.001        0.177     0.230   not a pivot   (pooled var would have fooled you!)
baking                 0.188        0.188     0.250   PIVOT (choice matters)
plating                0.000        0.000     1.000   not a pivot
```

This is the credit-assignment trap made concrete: **the naive signal flags prep (0.177); the correct signal rejects it (0.001) and finds baking (0.188).** With `--pivot-tau 0.05`, only baking survives.

## Results (real run, Qwen2.5-0.5B, single T4)

Real run of `databricks_launcher` (`epochs=3, G=4, M=16, pivot-tau=0.05, beta=0.02, lr=1e-5`):

| Metric | Base π₀ | SFT | E2E GRPO | **PivotRL (discovered)** |
|---|---|---|---|---|
| Discovered pivot | — | — | — | **baking** ✓ |
| Offline discovery rollouts | 0 | 0 | 0 | 192 |
| Training rollout-turns | 0 | 0 | 36 | **12** |
| OOD drift `KL(π_θ‖π₀)` ↓ | 0.000 | 1.100 | 0.546 | **0.213** |
| P(rescue) at pivot ↑ | 0.253 | 0.926 | 0.898 | 0.911 |
| ↳ lift over base | — | +0.672 | +0.645 | **+0.658** |

Three things this confirms on a real model (not just the toy math above):

1. **Discovery works from the end-only reward.** PivotRL correctly discovered `baking` as the sole pivot — no per-turn rewards required.
2. **The model genuinely learns the pivot.** `P(rescue)` rises from a near-random **0.253** (untrained base) to **0.911** (+0.658) — real learning under a terminal signal, matching what SFT/E2E achieve.
3. **Same PivotRL payoff as the dense demo:** **3× fewer training rollout-turns** than end-to-end GRPO (12 vs 36) and the **least OOD drift** (0.213 vs SFT's 1.100).

**The scale caveat, made concrete:** discovery cost **192** offline rollouts here, so PivotRL's *total* (192 + 12) is larger than E2E's 36 *at this toy scale* — the 16-turn training saving can't outrun a fixed 192-rollout probe over just 3 epochs. Discovery is a **one-time, offline** cost; the per-step 3× training saving is what compounds over a real (thousands-of-steps) run, where the fixed probe becomes negligible. The win is a scale story, not a 3-epoch-toy story.

## The honest cost

Discovery isn't free — it spends **offline completion-rollouts** (here `3 turns × 4 candidates × M`). That's the price of not having per-turn rewards. It's a **one-time, offline** cost, after which training concentrates on the single discovered pivot (vs. end-to-end GRPO grinding through all three turns). This is exactly the trade PivotRL makes in the real world: pay a bounded profiling cost up front to avoid expensive full-trajectory RL on turns that teach nothing.

## Files

```
terminal-reward/
├── kitchen_env.py            # 3-turn env with terminal reward ONLY (terminal_reward())
├── train_terminal_grpo.py    # discover_pivots() + turn-level GRPO; Base/SFT/E2E/PivotRL arms
└── databricks_launcher.py    # Databricks notebook runner (JSON summary)
```

## Run

```bash
pip install -r ../requirements.txt

python kitchen_env.py                       # env self-test (terminal reward)
python train_terminal_grpo.py --epochs 3 --group-size 4 --m 16
```

Key flags: `--m` (completion-rollouts per candidate during discovery), `--pivot-tau` (min action-value variance to call a turn a pivot), plus the usual `--group-size`, `--beta`, `--lr`, `--epochs`, `--mode {all,S,A,B}`, `--device`.

On Databricks: import this folder and run `databricks_launcher` on a single-node T4 (latest LTS ML GPU). Same import gotcha as the other folder — bring in `kitchen_env.py`/`train_terminal_grpo.py` as workspace files (`--format RAW`, keep the `.py`), and `databricks_launcher` as a notebook (`--format SOURCE --language PYTHON`).

## Caveat

Same pedagogical caveats as the dense demo (tiny model, fixed candidate action pools, illustrative OOD projection). The terminal variant is *more* faithful — it makes discovery necessary and shows the credit-assignment trap — but it still assumes a working verifier (`terminal_reward`). In genuinely open-ended domains, that verifier is the hard part; PivotRL makes training cheaper *given* one, it doesn't remove the need for it.
