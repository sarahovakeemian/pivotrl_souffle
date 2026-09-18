"""
souffle_simulator.py
====================

A small **executable soufflé model** used as a *real* functional verifier.

Why this exists
---------------
In `../dense-reward/`, the per-turn reward is a keyword lookup
(`if "tent" in action: good`). That's a disguised oracle — a human hand-labels
which strings are good, which is exactly the cheat PivotRL avoids in practice.

Here the reward instead comes from **simulate-and-compare**, mirroring how a
real functional verifier works (run the action, check the effect; compare to
the expert action's effect — cf. NeMo/Gym's tool-call *argument comparison*):

  1. Actions are **structured**: each candidate is an utterance PLUS the typed
     `effect` a tool-schema/parser would emit (this is the "verifiable action
     space" assumption — see `parse_effect` for the NL fallback).
  2. The soufflé has **state** (`SouffleState`); each effect *mutates* it via
     simple physics-ish dynamics (`apply`).
  3. `functional_verifier(turn, a)` runs the candidate AND the expert action
     from the same state and returns 1.0 iff they reach an **equivalent
     outcome** (`outcome_signature`). No keyword says "good" — the verdict
     falls out of the simulated dynamics.

This means a *novel* action is judged by its simulated effect, and two
differently-worded actions with the same effect both score 1.0 (functional
equivalence), while exact string-match would wrongly reject the paraphrase.

The honest boundary: something must still map free-form English → a structured
`effect`. Real systems buy that with a **structured action space** (tool calls)
or a **learned parser/judge**. Here the candidate pool ships structured effects
directly (the honest, tool-call-like path); `parse_effect` is a small keyword
NL fallback for arbitrary text, explicitly the piece you'd replace with an LLM.

No third-party dependencies; run `python souffle_simulator.py` for the self-test.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Turns
# --------------------------------------------------------------------------- #
TURN_PREP = "preparation"
TURN_BAKE = "baking"
TURN_PLATE = "plating"
TURNS: List[str] = [TURN_PREP, TURN_BAKE, TURN_PLATE]


# --------------------------------------------------------------------------- #
# Soufflé state + dynamics
# --------------------------------------------------------------------------- #
@dataclass
class SouffleState:
    """The soufflé's physical state, mutated turn by turn."""
    greased: bool = False
    coated: bool = False
    prep_ok: bool = False
    oven_temp: float = 0.9   # already spiking as we enter the bake turn
    browning: float = 0.3    # 0 pale · ~0.5 golden · >0.85 burnt
    rise: float = 0.0        # how high it has climbed (0..1)
    structure: float = 1.0   # structural integrity (1 intact · <0.5 collapsed)
    sweet_finish: bool = False


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def apply(turn: str, effect: Dict, state: SouffleState) -> SouffleState:
    """Apply one structured effect to the soufflé state and return the new state.

    This is the *dynamics* — the source of truth for reward. No string here
    decides "good"; outcomes emerge from the numbers.
    """
    s = replace(state)  # copy; never mutate the caller's state

    if turn == TURN_PREP:
        # Grease lets it release; a coating gives the batter something to climb.
        if effect.get("grease"):
            s.greased = True
        if effect.get("coat"):
            s.coated = True
        s.prep_ok = s.greased and s.coated

    elif turn == TURN_BAKE:
        # The danger: the oven is spiking and the top is browning too fast.
        if effect.get("vent"):
            # Opening the door = thermal shock -> it collapses.
            s.structure = 0.0
            s.rise = min(s.rise, 0.2)
            return s
        s.oven_temp = _clamp(s.oven_temp + float(effect.get("temp_delta", 0.0)))
        # Covering shields the top; otherwise browning tracks the (high) temp.
        shield = 0.3 if effect.get("cover") else 1.0
        s.browning = _clamp(s.browning + s.oven_temp * shield)
        # It climbs only if the ramekin was prepped and it hasn't been shocked,
        # and only if the top isn't being blasted (covered, or the temp backed off).
        if s.prep_ok and s.structure >= 0.5:
            s.rise = 0.9 if (effect.get("cover") or s.oven_temp < 0.6) else 0.5

    elif turn == TURN_PLATE:
        s.sweet_finish = bool(effect.get("sweet"))

    return s


def outcome_signature(s: SouffleState) -> Tuple:
    """The outcome-relevant projection of state used for equivalence.

    Two states are 'functionally equivalent' iff these categorical outcomes
    match — not iff their raw floats are identical. (So two rescues that leave
    the soufflé at browning 0.45 vs 0.70 are still equivalent: both golden.)
    """
    return (
        s.prep_ok,             # ramekin ready?
        s.structure < 0.5,     # collapsed?
        s.browning > 0.85,     # burnt?
        s.rise > 0.8,          # risen?
        s.sweet_finish,        # sweet finish applied?
    )


# --------------------------------------------------------------------------- #
# Structured-effect action catalog  (utterance + typed effect)
# --------------------------------------------------------------------------- #
@dataclass
class Action:
    text: str
    effect: Dict
    is_expert: bool = False


# Ordered so the bake pool interleaves rescues/failures (a small sampled group
# still spans both outcomes AND keeps mean < lambda_diff -> stays a pivot).
ACTIONS: Dict[str, List[Action]] = {
    TURN_PREP: [
        Action("Grease the ramekins and dust them with sugar", {"grease": True, "coat": True}, is_expert=True),
        Action("Butter the ramekins and coat with a fine layer of sugar", {"grease": True, "coat": True}),
        Action("Grease thoroughly, then dust with flour", {"grease": True, "coat": True}),
        Action("Brush with melted butter and dust with cocoa", {"grease": True, "coat": True}),
        Action("Chill the greased, sugared ramekins, then fill", {"grease": True, "coat": True}),
    ],
    TURN_BAKE: [
        Action("Tent the soufflé with foil and lower the temperature", {"cover": True, "temp_delta": -0.4}, is_expert=True),
        Action("Open the oven door to cool it down", {"vent": True}),
        Action("Loosely cover the top with foil and reduce the heat", {"cover": True, "temp_delta": -0.3}),
        Action("Increase the heat to set the top faster", {"temp_delta": 0.3}),
        Action("Move it to a lower rack and turn the temperature down", {"cover": False, "temp_delta": -0.5}),
        Action("Crank the broiler for a golden top", {"temp_delta": 0.5}),
        Action("Leave it and hope for the best", {"temp_delta": 0.0}),
    ],
    TURN_PLATE: [
        Action("Dust with powdered sugar", {"sweet": True}, is_expert=True),
        Action("Drizzle with warm chocolate syrup", {"sweet": True}),
        Action("Sprinkle with cocoa and a little sugar", {"sweet": True}),
        Action("Spoon over a berry compote", {"sweet": True}),
        Action("Finish with a caramel drizzle", {"sweet": True}),
        Action("Top with whipped cream and shaved chocolate", {"sweet": True}),
    ],
}


def candidate_texts(turn: str) -> List[str]:
    return [a.text for a in ACTIONS[turn]]


def expert_action(turn: str) -> str:
    return next(a.text for a in ACTIONS[turn] if a.is_expert)


def expert_actions() -> Dict[str, str]:
    return {t: expert_action(t) for t in TURNS}


# --------------------------------------------------------------------------- #
# NL -> structured effect
# --------------------------------------------------------------------------- #
def _known_effect(turn: str, text: str):
    for a in ACTIONS[turn]:
        if a.text == text:
            return a.effect
    return None


def parse_effect(turn: str, text: str) -> Dict:
    """Map a free-form English action to a structured effect.

    Primary path: exact catalog lookup (actions ship structured effects, like
    tool calls). Fallback: a small keyword parser for arbitrary text -- this is
    the ONE place a keyword heuristic lives, and it is explicitly the component
    you'd replace with an LLM/tool-schema parser in a real system. It only runs
    for text that isn't a catalogued structured action.
    """
    known = _known_effect(turn, text)
    if known is not None:
        return known

    a = (text or "").lower()
    if turn == TURN_PREP:
        grease = any(w in a for w in ("grease", "butter", "brush", "oil"))
        coat = any(w in a for w in ("sugar", "flour", "cocoa", "dust", "coat"))
        return {"grease": grease, "coat": coat}
    if turn == TURN_BAKE:
        if ("open" in a and ("door" in a or "oven" in a)):
            return {"vent": True}
        cover = any(w in a for w in ("tent", "foil", "cover", "shield"))
        if any(w in a for w in ("lower", "reduce", "down", "drop", "cool")):
            temp = -0.4
        elif any(w in a for w in ("increase", "raise", "crank", "higher", "hotter", "broil")):
            temp = 0.4
        else:
            temp = 0.0
        return {"cover": cover, "temp_delta": temp}
    if turn == TURN_PLATE:
        sweet = any(w in a for w in (
            "sugar", "chocolate", "cocoa", "syrup", "caramel", "honey",
            "berry", "berries", "whipped cream", "sweet",
        ))
        return {"sweet": sweet}
    return {}


# --------------------------------------------------------------------------- #
# The functional verifier: simulate-and-compare to the expert action
# --------------------------------------------------------------------------- #
def state_before(turn: str) -> SouffleState:
    """State entering `turn`, with all earlier turns taken by the expert.

    (States come from the SFT trajectory -- earlier turns are the demonstrated
    expert actions, exactly as PivotRL profiles a state.)
    """
    s = SouffleState()
    for t in TURNS:
        if t == turn:
            break
        s = apply(t, parse_effect(t, expert_action(t)), s)
    return s


def functional_verifier(turn: str, candidate_text: str, expert_text: str | None = None) -> float:
    """r_func(s, a) = 1[ candidate reaches the same outcome as the expert ].

    Run the candidate AND the expert action from the same (expert-prefixed)
    state and compare their outcome signatures. This is the real thing:
    functionally-equivalent actions score 1.0 even if worded differently;
    exact string match would reject the paraphrase.
    """
    expert_text = expert_text or expert_action(turn)
    s0 = state_before(turn)
    s_cand = apply(turn, parse_effect(turn, candidate_text), s0)
    s_exp = apply(turn, parse_effect(turn, expert_text), s0)
    return 1.0 if outcome_signature(s_cand) == outcome_signature(s_exp) else 0.0


def exact_match_reward(turn: str, candidate_text: str) -> float:
    """Naive baseline verifier: 1.0 only if the string equals the expert's.

    Included to *show* why functional verification matters -- it wrongly
    scores valid paraphrases 0.0.
    """
    return 1.0 if candidate_text.strip() == expert_action(turn).strip() else 0.0


# --------------------------------------------------------------------------- #
# Whole-trajectory rollout (for narration / self-test)
# --------------------------------------------------------------------------- #
def run_trajectory(actions_by_turn: Dict[str, str]) -> Tuple[SouffleState, float]:
    """Run all three turns; return final state and a 0/1 'good soufflé' quality."""
    s = SouffleState()
    for t in TURNS:
        s = apply(t, parse_effect(t, actions_by_turn[t]), s)
    good = s.prep_ok and s.structure >= 0.5 and s.browning <= 0.85 and s.rise > 0.8 and s.sweet_finish
    return s, (1.0 if good else 0.0)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    print("=" * 72)
    print("  souffle_simulator self-test")
    print("=" * 72)

    exp = expert_actions()

    # 1) Expert full run -> a good soufflé.
    _, q = run_trajectory(exp)
    print(f"[expert run] final quality = {q:.1f}  (expected 1.0)")
    assert q == 1.0

    # 2) Per-turn functional rewards via simulate-and-compare.
    print("\n[per-turn functional_verifier rewards]")
    for turn in TURNS:
        print(f"  {turn}:")
        for a in candidate_texts(turn):
            r = functional_verifier(turn, a)
            tag = "expert" if a == expert_action(turn) else ""
            print(f"    {r:.1f}  {a}  {tag}")

    # 3) Functional equivalence: a differently-worded rescue scores 1.0, and
    #    the exact-match baseline wrongly scores it 0.0.
    print("\n[functional vs exact-match on a paraphrased rescue]")
    para = "Loosely cover the top with foil and reduce the heat"  # not the expert string
    f = functional_verifier(TURN_BAKE, para)
    e = exact_match_reward(TURN_BAKE, para)
    print(f"  action: {para!r}")
    print(f"    functional_verifier = {f:.1f}  (expected 1.0 -- same effect as expert)")
    print(f"    exact_match_reward  = {e:.1f}  (expected 0.0 -- wrongly punishes paraphrase)")
    assert f == 1.0 and e == 0.0

    # 4) A brand-new action never catalogued, judged purely by simulated effect.
    novel = "Slide an empty tray onto the rack above to diffuse the heat, and ease the temperature down"
    rn = functional_verifier(TURN_BAKE, novel)
    print(f"\n[novel uncatalogued action] {novel!r}\n    functional_verifier = {rn:.1f} (judged by effect, no keyword list)")

    # 5) Pivot structure: bake mixed, prep/plate uniform.
    print("\n[reward distribution per turn over the full catalog]")
    import statistics
    for turn in TURNS:
        rs = [functional_verifier(turn, a) for a in candidate_texts(turn)]
        print(f"  {turn:<12} rewards={rs}  mean={statistics.fmean(rs):.3f}  var={statistics.pvariance(rs):.4f}")
    bake_rs = [functional_verifier(TURN_BAKE, a) for a in candidate_texts(TURN_BAKE)]
    assert statistics.pvariance(bake_rs) > 0.0, "baking must be mixed (a pivot)"
    for turn in (TURN_PREP, TURN_PLATE):
        rs = [functional_verifier(turn, a) for a in candidate_texts(turn)]
        assert statistics.pvariance(rs) == 0.0, f"{turn} must be uniform (not a pivot)"

    print("\n  All simulator self-tests PASSED.")
    print("=" * 72)


if __name__ == "__main__":
    _selftest()
