# 🍽️ Dense-Reward PivotRL — with a *real* functional verifier (simulator)

This is the **faithful** version of the dense-reward demo. See [`../dense-reward/`](../dense-reward) for the simpler one and [`../terminal-reward/`](../terminal-reward) for the discovery variant.

## Why this folder exists

`../dense-reward/` computes each turn's reward with a **keyword lookup** (`if "tent" in action: good`). That's honest for teaching the *mechanism*, but it's a disguised oracle — a human hand-labels which strings are good. Real PivotRL doesn't do that. Its reward is `r_func(s,a) = 1[a ∈ M(s)]`, where `M(s)` is **derived from the SFT expert action** by a **functional verifier** (e.g. NeMo/Gym's tool-call *argument comparison*).

This folder replaces the keyword oracle with **simulate-and-compare**:

```
reward(s, a) = 1  iff  running action a  reaches the same OUTCOME as
                       running the expert action a*   (both from state s)
```

Nothing says "good" — the verdict falls out of an executable soufflé model.

## How the verifier works (`souffle_simulator.py`)

1. **Structured-effect actions.** Every candidate is an utterance **plus** the typed `effect` a tool-schema/parser would emit — e.g. `"Tent the soufflé with foil and lower the temperature"` → `{cover: True, temp_delta: -0.4}`. (This is the "verifiable action space" assumption; `parse_effect` is a small NL fallback, explicitly the piece you'd swap for an LLM/tool-schema.)
2. **A stateful soufflé.** `SouffleState` tracks `rise, structure, browning, oven_temp, prep_ok, sweet_finish`. Each effect **mutates** it via simple physics-ish dynamics (`apply`): covering shields the top, opening the door collapses it, adding heat burns it, etc.
3. **Simulate-and-compare.** `functional_verifier(turn, a)` runs both the candidate and the expert action from the same state and returns `1.0` iff they land on the same **outcome signature** (`prep_ok, collapsed, burnt, risen, sweet`).

Three things this buys you that the keyword version can't show:

- **Functional equivalence is real, not listed.** `"Loosely cover the top with foil and reduce the heat"` scores **1.0** (same simulated outcome as the expert), while `exact_match_reward` scores it **0.0** — the `ls` vs `ls -l` lesson, made visible.
- **Novel actions are judged by effect.** An uncatalogued action like *"slide a tray onto the rack above to diffuse the heat and ease the temperature down"* is scored by its simulated result, with no keyword list.
- **The pivot structure is emergent, not asserted.** Rewards over the catalog:

  ```
  preparation  mean=1.000  var=0.000   (5 different prep methods, all functionally OK)
  baking       mean=0.429  var=0.245   (3 rescues vs 4 ruins  -> PIVOT)
  plating      mean=1.000  var=0.000   (6 different sweet finishes, all OK)
  ```

## Fleshed-out action catalog

Each turn has several options (structured effects in `souffle_simulator.ACTIONS`):

- **Preparation (5, all equivalent):** grease+sugar (expert), butter+sugar, grease+flour, butter+cocoa, chill-then-fill → all reach a ready ramekin → reward 1 (demonstrates functional equivalence).
- **Baking (7, mixed — the pivot):** 3 rescues (tent+lower [expert], loose-foil+reduce, lower-rack+down) vs 4 ruins (open door → collapse, increase heat → burn, broiler → burn, do nothing → burn). Interleaved so any sampled group spans both outcomes.
- **Plating (6, all sweet):** powdered sugar (expert), chocolate syrup, cocoa+sugar, berry compote, caramel, whipped cream + shaved chocolate → all sweet finishes → reward 1.

## Same pipeline, faithful reward

`kitchen_env.py` exposes the **exact same interface** as `../dense-reward/kitchen_env.py`, so `train_comparative_grpo.py` (Base / SFT / E2E-GRPO / PivotRL) and `databricks_launcher.py` run **unchanged** — only `get_functional_reward` now delegates to the simulator. Results match the dense demo; the reward's *provenance* is what changed.

```
dense-reward-simulator/
├── souffle_simulator.py       # the executable verifier (state, dynamics, actions, simulate-and-compare)
├── kitchen_env.py             # same env interface as ../dense-reward, reward -> simulator
├── train_comparative_grpo.py  # (unchanged copy) Base / SFT / E2E-GRPO / PivotRL
└── databricks_launcher.py     # Databricks notebook runner
```

## Run

```bash
pip install -r ../requirements.txt

python souffle_simulator.py     # verifier self-test (functional vs exact, novel action, pivot structure)
python kitchen_env.py           # env self-test (pivot filter picks baking)
python train_comparative_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3 --group-size 4
```

On Databricks: import all **three** `.py` modules as workspace files (`--format RAW`, keep the `.py`) plus `databricks_launcher` as a notebook (`--format SOURCE --language PYTHON`), on a single-node T4.

## The honest boundary

The simulator *is* the verifier you'd otherwise get from execution (run the code / the tool call) or a learned judge. Its realness bottlenecks on mapping free-form actions → checkable effects — which real systems solve with a **structured action space** (tool calls) or a **learned parser**, not a smarter keyword list. That boundary is the whole reason PivotRL is applied to verifiable domains (code, tool use, math) rather than open-ended prose.
