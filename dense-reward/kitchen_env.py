"""
kitchen_env.py
==============

A lightweight, self-contained text-agent environment: the **Gourmet Chef
Souffle Challenge**. It models a 3-turn agentic trajectory used to contrast
Traditional End-to-End GRPO against PivotRL (Turn-Level GRPO with Offline
Pivot Filtering).

Research grounding
------------------
GRPO (Group Relative Policy Optimization) was introduced in DeepSeekMath
(arXiv:2402.03300). It normalizes each candidate's reward against the mean and
standard deviation of its sampled group:

        A_i = (r_i - mean(r)) / (std(r) + eps)

The key structural fact this environment demonstrates: when every action in a
group succeeds (all r = 1.0) or every action fails (all r = 0.0), the group
reward variance is 0, so *every* normalized advantage is 0, and the gradient
update is 0. Those turns burn rollout compute for no learning signal.

PivotRL (arXiv:2603.21383, "PivotRL: High Accuracy Agentic Post-Training at
Low Compute Cost", Yi et al., 2026) exploits this: it runs K local rollouts to
find "pivots" -- intermediate states with high reward variance (sigma^2 > 0)
and low reward mean (mu < lambda_diff) -- and concentrates training only there.
It also replaces strict string-matching rewards with domain verifiers that
award a functional reward r_func(s, a) = 1[a in M(s)], where M(s) is the set of
acceptable actions. The paper reports +4.17% in-domain, +10.04% OOD accuracy
over SFT, at 4x fewer rollout turns than end-to-end RL.

The 3 turns modeled here
------------------------
  Turn 1  Preparation  -- SFT-easy. Every action succeeds (r=1.0). Var = 0.
                          => zero advantage under GRPO. Pivot Filter discards.
  Turn 2  Baking       -- THE PIVOT. Mixed outcomes (rescue vs. collapse/burn).
                          High variance, low mean. Pivot Filter SELECTS this.
  Turn 3  Plating      -- Functional-verifier turn. Any sweet topping = r=1.0.
                          Easy, zero variance. Pivot Filter discards.

This module has no third-party dependencies and runs (and self-tests) with a
plain `python kitchen_env.py`.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List


# --------------------------------------------------------------------------- #
# Turn identifiers
# --------------------------------------------------------------------------- #
TURN_PREP = "preparation"
TURN_BAKE = "baking"
TURN_PLATE = "plating"
TURNS: List[str] = [TURN_PREP, TURN_BAKE, TURN_PLATE]

# The set of acceptable "sweet topping" tokens the Turn-3 functional verifier
# accepts. Any action containing one of these is functionally equivalent -- the
# verifier does NOT require an exact string match against the SFT action.
SWEET_TOPPINGS: List[str] = [
    "powdered sugar",
    "confectioner",
    "chocolate",
    "cocoa",
    "syrup",
    "caramel",
    "honey",
    "berries",
    "berry",
    "whipped cream",
    "sugar",
]


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
@dataclass
class GourmetChefEnv:
    """A 3-turn Gourmet Chef Souffle Challenge environment.

    The environment is deterministic given (state, action): rewards come from
    ``get_functional_reward``. Variance across a *group* of sampled actions is
    what drives (or fails to drive) the GRPO gradient -- see
    ``profile_turn`` / ``profile_all``.
    """

    turn_index: int = 0
    lambda_diff: float = 0.6  # PivotRL "low-mean" threshold for pivot selection.
    _done: bool = False

    # ---- state machine ---------------------------------------------------- #
    def reset(self) -> Dict:
        self.turn_index = 0
        self._done = False
        return self.state()

    @property
    def current_turn(self) -> str:
        return TURNS[min(self.turn_index, len(TURNS) - 1)]

    def state(self) -> Dict:
        """Return the current conversational state (prompt + turn metadata)."""
        return {
            "turn": self.current_turn,
            "turn_index": self.turn_index,
            "system": SYSTEM_PROMPT,
            "prompt": STATE_PROMPTS[self.current_turn],
            "done": self._done,
        }

    def step(self, action: str):
        """Advance one turn. Returns (next_state, reward, done)."""
        if self._done:
            raise RuntimeError("Episode already finished; call reset().")
        turn = self.current_turn
        reward = self.get_functional_reward(turn, action)
        self.turn_index += 1
        if self.turn_index >= len(TURNS):
            self._done = True
        return self.state(), reward, self._done

    # ---- reward / verifier ------------------------------------------------ #
    def get_functional_reward(self, state, action: str) -> float:
        """Return the raw functional reward (1.0 or 0.0) for an action.

        ``state`` may be a turn name (str) or a state dict from ``state()``.
        """
        turn = state["turn"] if isinstance(state, dict) else state
        a = (action or "").lower().strip()

        if turn == TURN_PREP:
            # Turn 1: any standard kitchen prep succeeds -> uniform 1.0.
            # (Var = 0 across the group => zero GRPO advantage.)
            return 1.0 if a else 0.0

        if turn == TURN_BAKE:
            # Turn 2 (PIVOT): a delicate rescue. Only tenting/covering AND/OR
            # backing off the heat saves the souffle. Opening the door or
            # adding heat collapses/burns it.
            rescues = ("tent" in a) or ("foil" in a) or ("cover" in a)
            adds_heat = any(w in a for w in ("increase", "raise", "crank", "higher", "hotter", "broil"))
            opens_door = ("open" in a) and ("door" in a or "oven" in a)
            if rescues and not adds_heat:
                return 1.0
            if opens_door or adds_heat:
                return 0.0
            # Any other fiddling with a rising souffle collapses it.
            return 0.0

        if turn == TURN_PLATE:
            # Turn 3: functional verifier. r_func = 1[action mentions a sweet
            # topping in the acceptable set M(s)]. No exact-match required.
            return 1.0 if any(t in a for t in SWEET_TOPPINGS) else 0.0

        return 0.0

    # ---- rollout candidate pools ------------------------------------------ #
    def candidate_actions(self, turn) -> List[str]:
        """Return the pool of candidate actions a policy might sample at a turn.

        Used by the training script to draw a group of G (or K) rollouts.
        """
        turn = turn["turn"] if isinstance(turn, dict) else turn
        return list(CANDIDATE_ACTIONS[turn])

    def rollout_group(self, turn, n: int) -> List[str]:
        """Return a group of ``n`` sampled rollouts for a turn.

        Cycles the candidate pool so every reward outcome stays represented
        regardless of ``n`` (models drawing ``n`` on-policy samples with
        replacement). Guarantees ``len(result) == max(1, n)`` so rollout-turn
        counts and group variance are correct for any G or K -- including
        n > pool size (genuine repeats) and n < pool size (still spans outcomes,
        since the rescue action is listed first for the pivot turn).
        """
        pool = self.candidate_actions(turn)
        n = max(1, int(n))
        return [pool[i % len(pool)] for i in range(n)]

    # ---- pivot profiling -------------------------------------------------- #
    def profile_turn(self, turn, sampler: "Callable[[str], List[str]] | None" = None) -> Dict:
        """Profile a turn's reward statistics over its candidate group.

        Returns mean, (population) variance, std, whether the group carries a
        GRPO signal (variance > 0), and whether PivotRL would select it as a
        pivot (variance > 0 AND mean < lambda_diff).
        """
        turn = turn["turn"] if isinstance(turn, dict) else turn
        actions = sampler(turn) if sampler else self.candidate_actions(turn)
        rewards = [self.get_functional_reward(turn, a) for a in actions]
        mean = statistics.fmean(rewards)
        var = statistics.pvariance(rewards)  # population variance over the group
        std = var ** 0.5
        has_signal = var > 1e-9
        is_pivot = has_signal and (mean < self.lambda_diff)
        return {
            "turn": turn,
            "actions": actions,
            "rewards": rewards,
            "mean": mean,
            "variance": var,
            "std": std,
            "has_grpo_signal": has_signal,
            "is_pivot": is_pivot,
        }

    def profile_all(self, sampler: "Callable[[str], List[str]] | None" = None) -> Dict[str, Dict]:
        """Profile every turn. Keyed by turn name."""
        return {t: self.profile_turn(t, sampler=sampler) for t in TURNS}


# --------------------------------------------------------------------------- #
# Prompts / conversational states
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

# Candidate action pools (a group of 4) for each turn. Chosen so that:
#   prep  -> all 1.0  (variance 0, no GRPO signal)
#   bake  -> mixed    (variance > 0, low mean -> PIVOT)
#   plate -> all 1.0  (variance 0, no GRPO signal)  -- but functionally varied.
CANDIDATE_ACTIONS: Dict[str, List[str]] = {
    TURN_PREP: [
        "Grease the ramekins and dust them with sugar",
        "Butter the ramekins thoroughly",
        "Coat the ramekins with a fine layer of sugar",
        "Gather and prep the ramekins for baking",
    ],
    TURN_BAKE: [
        # Rescue listed first so a small group (n < 4) still spans both reward
        # outcomes -> the pivot's variance survives truncation/cycling.
        "Tent the souffle with foil and lower the temp",   # -> rises!    (1.0)
        "Open the oven door to cool it down",              # -> collapses (0.0)
        "Increase the heat to brown the top faster",       # -> burns     (0.0)
        "Crank the broiler to set the top",                # -> burns     (0.0)
    ],
    TURN_PLATE: [
        "Dust with powdered sugar",                        # -> 1.0
        "Drizzle with chocolate syrup",                    # -> 1.0
        "Sprinkle with cocoa",                             # -> 1.0
        "Finish with a dusting of confectioner's sugar",   # -> 1.0
    ],
}


# --------------------------------------------------------------------------- #
# Mock SFT trajectories
# --------------------------------------------------------------------------- #
def get_sft_trajectories() -> Dict:
    """Return a mock SFT trajectory dataset.

    Structure mirrors what an offline SFT corpus would provide: for each turn,
    the conversational state and the historically matched (expert) SFT action.
    PivotRL operates *on top of* these existing trajectories rather than
    generating fresh full-episode rollouts.
    """
    return {
        "task": "gourmet_chef_souffle_challenge",
        "system": SYSTEM_PROMPT,
        "num_turns": len(TURNS),
        "trajectory": [
            {
                "turn_index": 0,
                "turn": TURN_PREP,
                "state": STATE_PROMPTS[TURN_PREP],
                "sft_action": "Grease the ramekins and dust them with sugar",
                "note": "SFT-easy: all prep actions succeed (r=1.0).",
            },
            {
                "turn_index": 1,
                "turn": TURN_BAKE,
                "state": STATE_PROMPTS[TURN_BAKE],
                "sft_action": "Tent the souffle with foil and lower the temp",
                "note": "PIVOT: mixed outcomes; the only rescue among peers.",
            },
            {
                "turn_index": 2,
                "turn": TURN_PLATE,
                "state": STATE_PROMPTS[TURN_PLATE],
                "sft_action": "Dust with powdered sugar",
                "note": "Functional verifier: any sweet topping is acceptable.",
            },
        ],
    }


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    print("=" * 72)
    print("  GourmetChefEnv self-test")
    print("=" * 72)

    env = GourmetChefEnv()

    # 1) Canonical Turn-2 rollouts from the spec.
    canonical = {
        "Open oven door": 0.0,
        "Increase heat": 0.0,
        "Tent with foil and lower temp": 1.0,
    }
    print("\n[Turn 2 canonical rewards]")
    for action, expected in canonical.items():
        r = env.get_functional_reward(TURN_BAKE, action)
        flag = "OK" if abs(r - expected) < 1e-9 else "FAIL"
        print(f"  [{flag}] {action:<38} -> {r:.1f} (expected {expected:.1f})")
        assert abs(r - expected) < 1e-9, f"reward mismatch for {action!r}"

    # 2) Turn-3 functional verifier: syntactically different, all valid.
    print("\n[Turn 3 functional verifier -- all should be 1.0]")
    for action in ("Dust with powdered sugar", "Drizzle with chocolate syrup", "Sprinkle with cocoa"):
        r = env.get_functional_reward(TURN_PLATE, action)
        print(f"  [{'OK' if r == 1.0 else 'FAIL'}] {action:<38} -> {r:.1f}")
        assert r == 1.0
    bad = env.get_functional_reward(TURN_PLATE, "Pour gravy over it")
    print(f"  [{'OK' if bad == 0.0 else 'FAIL'}] {'Pour gravy over it':<38} -> {bad:.1f} (non-sweet -> 0.0)")
    assert bad == 0.0

    # 3) Per-turn profiling: which turns carry a GRPO signal / are pivots?
    print("\n[Pivot profiling over candidate groups (G=4)]")
    profiles = env.profile_all()
    for turn in TURNS:
        p = profiles[turn]
        print(
            f"  {turn:<12} mean={p['mean']:.3f}  var={p['variance']:.4f}  "
            f"signal={str(p['has_grpo_signal']):<5}  pivot={p['is_pivot']}"
        )

    # Assertions encoding the demo's core claims.
    assert profiles[TURN_PREP]["variance"] == 0.0, "prep must have zero variance"
    assert profiles[TURN_PREP]["is_pivot"] is False
    assert profiles[TURN_PLATE]["variance"] == 0.0, "plating must have zero variance"
    assert profiles[TURN_PLATE]["is_pivot"] is False
    assert profiles[TURN_BAKE]["variance"] > 0.0, "baking must have positive variance"
    assert profiles[TURN_BAKE]["is_pivot"] is True, "baking must be selected as the pivot"

    pivots = [t for t in TURNS if profiles[t]["is_pivot"]]
    print(f"\n  Pivot Filter selects: {pivots}  (expected: ['{TURN_BAKE}'])")
    assert pivots == [TURN_BAKE]

    # 4) Full episode walkthrough with the SFT expert actions.
    print("\n[Episode walkthrough with SFT expert actions]")
    sft = get_sft_trajectories()
    env.reset()
    total = 0.0
    for step in sft["trajectory"]:
        _, r, done = env.step(step["sft_action"])
        total += r
        print(f"  {step['turn']:<12} action={step['sft_action']!r:<50} r={r:.1f}")
    print(f"  Episode return = {total:.1f} / {len(TURNS)}  (done={done})")
    assert total == float(len(TURNS)), "SFT expert should score a perfect episode"

    print("\n" + "=" * 72)
    print("  All self-tests PASSED.")
    print("=" * 72)


if __name__ == "__main__":
    _selftest()
