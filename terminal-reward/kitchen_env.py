"""
kitchen_env.py  (TERMINAL-REWARD variant)
==========================================

Same Gourmet Chef Souffle Challenge as the dense-reward demo, but with the
reward structure real agentic tasks actually have: **there is no per-turn
score.** You only find out whether the soufflé is good or bad **at the very
end**, once all three actions (prep -> bake -> plate) are done.

Why this variant exists
-----------------------
The dense-reward demo hands out a reward at every turn, which makes the pivot
(the high-variance turn) trivially visible -- we basically tell the algorithm
where to train. That's great for teaching the *mechanism*, but it sidesteps the
hard part of PivotRL: in real tasks (code that only passes tests at the end,
math graded on the final answer, a web search judged only by whether you found
the answer) the environment gives a single **terminal** reward, and you must
*discover* which turns were pivotal.

This module exposes ONLY a terminal reward:

    terminal_reward(actions) -> 1.0 if the finished souffle is good else 0.0

"Good" depends on the whole run, but is dominated by the baking decision:
  * Prep   -- any reasonable prep is fine.
  * Bake   -- only "tent + lower the temperature" saves it; opening the door or
              adding heat ruins it. (THE decisive step.)
  * Plate  -- any sweet topping is fine.

So the finished soufflé is good iff every step was acceptable -- which, because
prep and plating are forgiving, hinges almost entirely on the baking choice.
The pivot-discovery code (see train_terminal_grpo.py) has to *figure that out*
from the terminal signal alone, by rolling out to completion and watching which
turn's choice actually swings the final verdict.

No third-party dependencies; run `python kitchen_env.py` for the self-test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# --------------------------------------------------------------------------- #
# Turn identifiers
# --------------------------------------------------------------------------- #
TURN_PREP = "preparation"
TURN_BAKE = "baking"
TURN_PLATE = "plating"
TURNS: List[str] = [TURN_PREP, TURN_BAKE, TURN_PLATE]

SWEET_TOPPINGS: List[str] = [
    "powdered sugar", "confectioner", "chocolate", "cocoa", "syrup",
    "caramel", "honey", "berries", "berry", "whipped cream", "sugar",
]


# --------------------------------------------------------------------------- #
# Environment (terminal reward only)
# --------------------------------------------------------------------------- #
@dataclass
class GourmetChefTerminalEnv:
    """A 3-turn souffle challenge whose reward arrives ONLY at the end."""

    lambda_diff: float = 0.6  # "low-mean" threshold reused by the pivot filter

    # ---- per-step acceptability (INTERNAL -- not a training signal) -------- #
    def _acceptable(self, turn: str, action: str) -> bool:
        """Whether one action is acceptable at a turn. Used only to *compose*
        the terminal reward; the trainer never sees this directly."""
        a = (action or "").lower().strip()
        if turn == TURN_PREP:
            return bool(a)  # any real prep works
        if turn == TURN_BAKE:
            rescues = ("tent" in a) or ("foil" in a) or ("cover" in a)
            adds_heat = any(w in a for w in ("increase", "raise", "crank", "higher", "hotter", "broil"))
            return rescues and not adds_heat  # only backing off the heat saves it
        if turn == TURN_PLATE:
            return any(t in a for t in SWEET_TOPPINGS)
        return False

    # ---- the ONLY public reward: terminal ---------------------------------- #
    def terminal_reward(self, actions) -> float:
        """Reward for a COMPLETED trajectory of 3 actions (prep, bake, plate).

        ``actions`` may be a list/tuple in turn order, or a dict keyed by turn.
        Returns 1.0 iff the finished souffle is good (every step acceptable),
        else 0.0. There is no intermediate/per-turn reward.
        """
        if isinstance(actions, dict):
            seq = [actions[t] for t in TURNS]
        else:
            seq = list(actions)
        if len(seq) != len(TURNS):
            raise ValueError(f"terminal_reward expects {len(TURNS)} actions, got {len(seq)}")
        return 1.0 if all(self._acceptable(t, a) for t, a in zip(TURNS, seq)) else 0.0

    # ---- rollout candidate pools ------------------------------------------ #
    def candidate_actions(self, turn) -> List[str]:
        turn = turn["turn"] if isinstance(turn, dict) else turn
        return list(CANDIDATE_ACTIONS[turn])

    def rollout_group(self, turn, n: int) -> List[str]:
        """A group of n candidate actions, cycling the pool so all reward
        outcomes stay represented for any n (rescue is listed first)."""
        pool = self.candidate_actions(turn)
        n = max(1, int(n))
        return [pool[i % len(pool)] for i in range(n)]


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

# Rescue listed first at the bake turn so small groups still span both outcomes.
CANDIDATE_ACTIONS: Dict[str, List[str]] = {
    TURN_PREP: [
        "Grease the ramekins and dust them with sugar",
        "Butter the ramekins thoroughly",
        "Coat the ramekins with a fine layer of sugar",
        "Gather and prep the ramekins for baking",
    ],
    TURN_BAKE: [
        "Tent the souffle with foil and lower the temp",   # -> good ending
        "Open the oven door to cool it down",              # -> collapses
        "Increase the heat to brown the top faster",       # -> burns
        "Crank the broiler to set the top",                # -> burns
    ],
    TURN_PLATE: [
        "Dust with powdered sugar",
        "Drizzle with chocolate syrup",
        "Sprinkle with cocoa",
        "Finish with a dusting of confectioner's sugar",
    ],
}


def get_sft_trajectories() -> Dict:
    """Mock SFT trajectory (expert actions). The expert run scores a terminal
    reward of 1.0 (every step acceptable)."""
    return {
        "task": "gourmet_chef_souffle_challenge_terminal",
        "system": SYSTEM_PROMPT,
        "num_turns": len(TURNS),
        "trajectory": [
            {"turn_index": 0, "turn": TURN_PREP, "state": STATE_PROMPTS[TURN_PREP],
             "sft_action": "Grease the ramekins and dust them with sugar"},
            {"turn_index": 1, "turn": TURN_BAKE, "state": STATE_PROMPTS[TURN_BAKE],
             "sft_action": "Tent the souffle with foil and lower the temp"},
            {"turn_index": 2, "turn": TURN_PLATE, "state": STATE_PROMPTS[TURN_PLATE],
             "sft_action": "Dust with powdered sugar"},
        ],
    }


def expert_actions() -> Dict[str, str]:
    """Convenience: {turn: expert_action} from the SFT trajectory."""
    return {step["turn"]: step["sft_action"] for step in get_sft_trajectories()["trajectory"]}


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    print("=" * 72)
    print("  GourmetChefTerminalEnv self-test")
    print("=" * 72)
    env = GourmetChefTerminalEnv()
    exp = expert_actions()

    # Expert full run -> good souffle (reward 1.0).
    r = env.terminal_reward(exp)
    print(f"[expert run] terminal_reward = {r:.1f}  (expected 1.0)")
    assert r == 1.0

    # Ruin exactly one turn at a time; only the baking ruin should matter given
    # the others stay expert.
    bad_bake = dict(exp); bad_bake[TURN_BAKE] = "Open the oven door to cool it down"
    print(f"[bad bake ] terminal_reward = {env.terminal_reward(bad_bake):.1f}  (expected 0.0)")
    assert env.terminal_reward(bad_bake) == 0.0

    # A non-sweet plating also ruins the terminal reward.
    bad_plate = dict(exp); bad_plate[TURN_PLATE] = "Pour gravy over it"
    print(f"[bad plate] terminal_reward = {env.terminal_reward(bad_plate):.1f}  (expected 0.0)")
    assert env.terminal_reward(bad_plate) == 0.0

    # Different-but-acceptable prep + plate, correct bake -> still good.
    variant = {TURN_PREP: "Butter the ramekins thoroughly",
               TURN_BAKE: "Tent the souffle with foil and lower the temp",
               TURN_PLATE: "Sprinkle with cocoa"}
    print(f"[variant  ] terminal_reward = {env.terminal_reward(variant):.1f}  (expected 1.0)")
    assert env.terminal_reward(variant) == 1.0

    print("\n  All terminal-env self-tests PASSED.")
    print("=" * 72)


if __name__ == "__main__":
    _selftest()
