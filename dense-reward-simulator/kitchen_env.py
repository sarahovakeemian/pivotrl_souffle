"""
kitchen_env.py  (SIMULATOR-VERIFIER variant)
=============================================

Same environment interface as `../dense-reward/kitchen_env.py`, so the exact
same `train_comparative_grpo.py` runs on top of it unchanged. The ONLY
difference is where the per-turn reward comes from:

  dense-reward/          get_functional_reward = keyword lookup (a hand oracle)
  dense-reward-simulator/  get_functional_reward = simulate-and-compare against
                           the SFT expert action (souffle_simulator.functional_verifier)

So the reward now genuinely flows: SFT expert action (from the trajectory) →
functional verifier (an executable soufflé simulator) → per-turn 0/1 reward.
The pivot structure is identical (baking = high variance; prep & plating =
zero variance), so all downstream results match — only the reward's provenance
becomes faithful to how PivotRL actually verifies actions.

Run `python kitchen_env.py` for the self-test.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Dict, List

import souffle_simulator as sim
from souffle_simulator import (
    TURNS,
    TURN_PREP,
    TURN_BAKE,
    TURN_PLATE,
    candidate_texts,
    expert_action,
    expert_actions,
    functional_verifier,
)

# Module-level catalog of candidate action strings per turn (utterances only).
CANDIDATE_ACTIONS: Dict[str, List[str]] = {t: candidate_texts(t) for t in TURNS}


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a world-class pastry chef executing a 3-step souffle challenge. "
    "At each step, respond with a single concise kitchen action."
)

STATE_PROMPTS: Dict[str, str] = {
    TURN_PREP: (
        "TURN 1 - PREPARATION. Your station is set. Prepare the ramekins so the "
        "souffle can climb: gather ramekins, grease them, and coat with sugar. "
        "What is your action?"
    ),
    TURN_BAKE: (
        "TURN 2 - BAKING. The souffle is rising beautifully in the oven -- but "
        "the temperature just spiked and the top is browning far too fast. It "
        "will collapse or burn within seconds. What is your action?"
    ),
    TURN_PLATE: (
        "TURN 3 - PLATING. The souffle is out of the oven, tall and proud. Apply "
        "a sweet finishing topping before it reaches the guest. What is your action?"
    ),
}


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
@dataclass
class GourmetChefEnv:
    """3-turn souffle env whose per-turn reward is produced by the simulator."""

    lambda_diff: float = 0.6

    # ---- reward: simulate-and-compare to the SFT expert action ------------ #
    def get_functional_reward(self, state, action: str) -> float:
        """r_func(s,a) via the executable soufflé verifier (not a keyword rule)."""
        turn = state["turn"] if isinstance(state, dict) else state
        return functional_verifier(turn, action)

    # ---- candidate pools -------------------------------------------------- #
    def candidate_actions(self, turn) -> List[str]:
        turn = turn["turn"] if isinstance(turn, dict) else turn
        return list(CANDIDATE_ACTIONS[turn])

    def rollout_group(self, turn, n: int) -> List[str]:
        """Group of n candidate actions, cycling the pool so all outcomes stay
        represented for any n (the bake pool is interleaved rescue/failure)."""
        pool = self.candidate_actions(turn)
        n = max(1, int(n))
        return [pool[i % len(pool)] for i in range(n)]

    # ---- pivot profiling -------------------------------------------------- #
    def profile_turn(self, turn, sampler: "Callable[[str], List[str]] | None" = None) -> Dict:
        turn = turn["turn"] if isinstance(turn, dict) else turn
        actions = sampler(turn) if sampler else self.candidate_actions(turn)
        rewards = [self.get_functional_reward(turn, a) for a in actions]
        mean = statistics.fmean(rewards)
        var = statistics.pvariance(rewards)
        has_signal = var > 1e-9
        return {
            "turn": turn,
            "actions": actions,
            "rewards": rewards,
            "mean": mean,
            "variance": var,
            "std": var ** 0.5,
            "has_grpo_signal": has_signal,
            "is_pivot": has_signal and (mean < self.lambda_diff),
        }

    def profile_all(self, sampler: "Callable[[str], List[str]] | None" = None) -> Dict[str, Dict]:
        return {t: self.profile_turn(t, sampler=sampler) for t in TURNS}


# --------------------------------------------------------------------------- #
# Mock SFT trajectories (expert actions come from the simulator catalog)
# --------------------------------------------------------------------------- #
def get_sft_trajectories() -> Dict:
    exp = expert_actions()
    return {
        "task": "gourmet_chef_souffle_challenge_simulated",
        "system": SYSTEM_PROMPT,
        "num_turns": len(TURNS),
        "trajectory": [
            {"turn_index": i, "turn": t, "state": STATE_PROMPTS[t], "sft_action": exp[t]}
            for i, t in enumerate(TURNS)
        ],
    }


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    print("=" * 72)
    print("  GourmetChefEnv (simulator-verifier) self-test")
    print("=" * 72)
    env = GourmetChefEnv()

    print("\n[pivot profiling over candidate groups]")
    profiles = env.profile_all()
    for t in TURNS:
        p = profiles[t]
        print(f"  {t:<12} mean={p['mean']:.3f} var={p['variance']:.4f} "
              f"signal={str(p['has_grpo_signal']):<5} pivot={p['is_pivot']}")

    assert profiles[TURN_PREP]["variance"] == 0.0 and not profiles[TURN_PREP]["is_pivot"]
    assert profiles[TURN_PLATE]["variance"] == 0.0 and not profiles[TURN_PLATE]["is_pivot"]
    assert profiles[TURN_BAKE]["variance"] > 0.0 and profiles[TURN_BAKE]["is_pivot"]
    pivots = [t for t in TURNS if profiles[t]["is_pivot"]]
    print(f"\n  Pivot Filter selects: {pivots}  (expected ['{TURN_BAKE}'])")
    assert pivots == [TURN_BAKE]

    # Rewards are produced by the simulator, and small sampled groups keep the
    # bake turn a pivot (mean < lambda_diff).
    for n in (2, 4, 6, 8):
        p = env.profile_turn(TURN_BAKE, sampler=lambda t: env.rollout_group(t, n))
        print(f"  bake group n={n}: mean={p['mean']:.3f} var={p['variance']:.4f} pivot={p['is_pivot']}")
        assert p["is_pivot"], f"bake should stay a pivot at n={n}"

    # SFT expert scores a perfect run.
    sft = get_sft_trajectories()
    total = sum(env.get_functional_reward(s["turn"], s["sft_action"]) for s in sft["trajectory"])
    print(f"\n  SFT expert episode reward = {total:.1f} / {len(TURNS)}")
    assert total == float(len(TURNS))

    print("\n  All env self-tests PASSED.")
    print("=" * 72)


if __name__ == "__main__":
    _selftest()
