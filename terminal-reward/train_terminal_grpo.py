"""
train_terminal_grpo.py  (TERMINAL-REWARD variant)
=================================================

The realistic version of the PivotRL soufflé demo: the environment gives a
reward **only at the end** (good soufflé = 1, ruined = 0), so the algorithm has
to *discover* which turn is the pivot instead of being told by per-turn rewards.

What's different from the dense-reward demo
-------------------------------------------
The star of this file is `discover_pivots()`. With no per-turn reward, we find
pivots by **rolling out to completion**:

  For each turn, freeze the earlier turns to the expert (SFT) actions, try each
  candidate action there, let the CURRENT policy finish the remaining turns, and
  score the finished soufflé. Repeat M times per candidate.

Then we compute TWO variance signals and compare them -- this is the teaching
moment:

  * NAIVE "pooled outcome variance": the spread of terminal rewards reachable
    from a state. This is the intuitive-but-WRONG signal: it also lights up on
    turns that merely sit UPSTREAM of a pivot (a soufflé started at prep can
    still be ruined later at baking), so it mis-flags prep as important.

  * CORRECT "action-value variance": how much your *choice at this turn* changes
    the expected final outcome (variance across candidates of their mean
    terminal reward). This isolates the real pivot: at prep every choice leads
    to the same downstream lottery (variance ~0), while at baking the choice
    swings the ending from 0 to 1 (variance high).

Once the pivot (baking) is discovered, training is the same turn-level GRPO as
the dense demo, using the terminal reward evaluated with expert context for the
non-pivot turns.

Arms: Base pi_0 (untrained) / SFT / End-to-End GRPO / PivotRL.

Run:  python train_terminal_grpo.py --epochs 3 --group-size 4 --m 16
"""

from __future__ import annotations

import argparse
import copy
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from kitchen_env import (
    GourmetChefTerminalEnv,
    TURNS,
    TURN_BAKE,
    SYSTEM_PROMPT,
    STATE_PROMPTS,
    get_sft_trajectories,
    expert_actions,
)

try:
    from tabulate import tabulate
    _HAVE_TABULATE = True
except Exception:  # pragma: no cover
    _HAVE_TABULATE = False


OOD_PROBE_PROMPTS: List[str] = [
    "What is the capital of France?",
    "Write a Python function that returns the nth Fibonacci number.",
    "Summarize the water cycle in one sentence.",
    "Translate 'good morning' into Spanish.",
]
PAPER_OOD_CEILING_PCT = 10.04


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_reference(model_name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] tokenizer + model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ref = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    ref.to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return tok, ref


def fresh_policy(ref):
    policy = copy.deepcopy(ref)
    policy.train()
    for p in policy.parameters():
        p.requires_grad_(True)
    return policy


# --------------------------------------------------------------------------- #
# Prompt / scoring helpers
# --------------------------------------------------------------------------- #
def build_prompt_text(tok, system: str, user: str) -> str:
    try:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"{system}\n\n{user}\n"


def action_token_logps(model, tok, prompt_text: str, action_text: str, device: str) -> torch.Tensor:
    prompt_ids = tok(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    action_ids = tok(action_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    input_ids = torch.cat([prompt_ids, action_ids], dim=1)
    logits = model(input_ids).logits
    logprobs = F.log_softmax(logits, dim=-1)
    P = prompt_ids.shape[1]
    A = action_ids.shape[1]
    pred_positions = torch.arange(P - 1, P - 1 + A, device=device)
    return logprobs[0, pred_positions, :].gather(-1, action_ids[0].unsqueeze(-1)).squeeze(-1)


def k3_kl(pol_logps: torch.Tensor, ref_logps: torch.Tensor) -> torch.Tensor:
    logr = ref_logps - pol_logps
    return (torch.exp(logr) - logr - 1.0).sum()


# --------------------------------------------------------------------------- #
# Policy's distribution over candidate actions (for on-policy continuations)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def policy_action_probs(policy, tok, turn, env, device) -> Tuple[List[str], List[float]]:
    """Length-normalized log-prob softmax over a turn's candidate pool = the
    policy's distribution over actions there."""
    prompt = build_prompt_text(tok, SYSTEM_PROMPT, STATE_PROMPTS[turn])
    actions = env.candidate_actions(turn)
    norm_logps = []
    for a in actions:
        lp = action_token_logps(policy, tok, prompt, a, device).sum()
        n_tok = max(1, len(tok(a, add_special_tokens=False).input_ids))
        norm_logps.append(float(lp) / n_tok)
    probs = torch.softmax(torch.tensor(norm_logps), dim=0).tolist()
    return actions, probs


def policy_distributions(policy, tok, env, device) -> Dict[str, Tuple[List[str], List[float]]]:
    """Cache the policy's action distribution for every turn (computed once)."""
    return {t: policy_action_probs(policy, tok, t, env, device) for t in TURNS}


def _sample(actions: List[str], probs: List[float], rng: random.Random) -> str:
    r = rng.random()
    c = 0.0
    for a, p in zip(actions, probs):
        c += p
        if r <= c:
            return a
    return actions[-1]


# --------------------------------------------------------------------------- #
# THE STAR: discover pivots from a terminal reward via rollout-to-completion
# --------------------------------------------------------------------------- #
@dataclass
class TurnProfile:
    turn: str
    action_value_var: float   # correct signal: does the CHOICE here move the outcome?
    pooled_outcome_var: float  # naive signal: raw spread of outcomes from this state (the trap)
    mean_reward: float
    is_pivot: bool
    n_rollouts: int


def discover_pivots(policy, tok, env, args, device, rng) -> Tuple[List[str], Dict[str, TurnProfile]]:
    """Find pivots from the terminal reward alone.

    States come from the SFT trajectory: at turn t, earlier turns are the expert
    actions; the candidate action goes at t; later turns are completed on-policy
    (sampled M times). We score the finished soufflé each time and compare the
    naive pooled-outcome variance against the correct action-value variance.
    """
    dists = policy_distributions(policy, tok, env, device)  # one-time model calls
    expert = expert_actions()
    report: Dict[str, TurnProfile] = {}
    total_rollouts = 0

    print(f"\n[discovery] rolling out to completion (M={args.m} per candidate) to find pivots...")
    print(f"[discovery] {'turn':<12} {'action-VALUE var':>16} {'pooled-outcome var':>19} {'mean R':>8}   verdict")
    for ti, turn in enumerate(TURNS):
        before = {TURNS[j]: expert[TURNS[j]] for j in range(ti)}
        after_turns = TURNS[ti + 1:]
        candidates = env.candidate_actions(turn)

        per_action_values: List[float] = []
        pooled: List[float] = []
        for a in candidates:
            rs: List[float] = []
            for _ in range(args.m):
                traj = dict(before)
                traj[turn] = a
                for at in after_turns:
                    acts, ps = dists[at]
                    traj[at] = _sample(acts, ps, rng)
                r = env.terminal_reward(traj)
                rs.append(r)
                pooled.append(r)
                total_rollouts += 1
            per_action_values.append(statistics.fmean(rs))

        av_var = statistics.pvariance(per_action_values)   # correct
        pool_var = statistics.pvariance(pooled)             # naive / trap
        mean_r = statistics.fmean(pooled)
        is_pivot = (av_var > args.pivot_tau) and (mean_r < env.lambda_diff)
        report[turn] = TurnProfile(turn, av_var, pool_var, mean_r, is_pivot, len(candidates) * args.m)

        verdict = "PIVOT (choice matters)" if is_pivot else "not a pivot"
        print(f"[discovery] {turn:<12} {av_var:>16.4f} {pool_var:>19.4f} {mean_r:>8.3f}   {verdict}")

    pivots = [t for t in TURNS if report[t].is_pivot]
    trap = [t for t in TURNS if (report[t].pooled_outcome_var > args.pivot_tau and not report[t].is_pivot)]
    print(f"[discovery] discovered pivots (action-value var > {args.pivot_tau}): {pivots}")
    if trap:
        print(f"[discovery] NOTE: naive pooled-outcome variance would ALSO have flagged {trap}")
        print(f"[discovery]       -> those turns only inherit variance from the downstream pivot;")
        print(f"[discovery]       action-value variance correctly rejects them. (credit-assignment trap)")
    print(f"[discovery] total offline completion-rollouts spent: {total_rollouts}")
    return pivots, report


# --------------------------------------------------------------------------- #
# Terminal reward for a single action (expert context for the other turns)
# --------------------------------------------------------------------------- #
def reward_for_action(env, expert: Dict[str, str], turn: str, action: str) -> float:
    """Terminal reward when `action` is taken at `turn` and every OTHER turn is
    the expert action. Isolates the action's contribution to the final verdict."""
    traj = dict(expert)
    traj[turn] = action
    return env.terminal_reward(traj)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def policy_pivot_confidence(policy, tok, env, device) -> float:
    """P(policy picks a *good* baking action) among the bake candidate group."""
    prompt = build_prompt_text(tok, SYSTEM_PROMPT, STATE_PROMPTS[TURN_BAKE])
    actions = env.candidate_actions(TURN_BAKE)
    norm_logps, good = [], []
    for a in actions:
        lp = action_token_logps(policy, tok, prompt, a, device).sum()
        n_tok = max(1, len(tok(a, add_special_tokens=False).input_ids))
        norm_logps.append(float(lp) / n_tok)
        good.append(env._acceptable(TURN_BAKE, a))
    probs = torch.softmax(torch.tensor(norm_logps), dim=0).tolist()
    return float(sum(p for p, g in zip(probs, good) if g))


@torch.no_grad()
def ood_drift(policy, ref, tok, device) -> float:
    kls = []
    for prompt in OOD_PROBE_PROMPTS:
        text = build_prompt_text(tok, "You are a helpful assistant.", prompt)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        pol = F.log_softmax(policy(ids).logits, dim=-1)
        rf = F.log_softmax(ref(ids).logits, dim=-1)
        kls.append(float((pol.exp() * (pol - rf)).sum(-1).mean()))
    return statistics.fmean(kls)


# --------------------------------------------------------------------------- #
# Turn-level GRPO loss (terminal reward w/ expert context)
# --------------------------------------------------------------------------- #
@dataclass
class TurnStats:
    turn: str
    mean: float
    std: float
    kl: float
    has_signal: bool


def grpo_group_loss(turn, policy, ref, tok, env, expert, group_size, beta, device):
    prompt_text = build_prompt_text(tok, SYSTEM_PROMPT, STATE_PROMPTS[turn])
    actions = env.rollout_group(turn, group_size)
    rewards = torch.tensor(
        [reward_for_action(env, expert, turn, a) for a in actions],
        dtype=torch.float32, device=device,
    )
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    advantages = torch.zeros_like(rewards) if std < 1e-8 else (rewards - mean) / (std + 1e-8)

    pg_terms, kl_terms = [], []
    for action, adv in zip(actions, advantages):
        pol_lp = action_token_logps(policy, tok, prompt_text, action, device)
        with torch.no_grad():
            ref_lp = action_token_logps(ref, tok, prompt_text, action, device)
        pg_terms.append(adv.detach() * pol_lp.sum())
        kl_terms.append(k3_kl(pol_lp, ref_lp))

    pg_loss = -torch.stack(pg_terms).mean()
    kl_loss = torch.stack(kl_terms).mean()
    loss = pg_loss + beta * kl_loss
    return loss, TurnStats(turn, float(mean), float(std), float(kl_loss.detach()), bool(std >= 1e-8))


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
@dataclass
class ModeResult:
    name: str
    trained_turns: List[str]
    discovery_rollouts: int
    train_rollout_turns: int
    epochs: int
    wall_clock_s: float
    ood_kl: float
    pivot_confidence: float
    discovered_pivots: List[str] = field(default_factory=list)


def run_baseline(ref, tok, env, args, device) -> ModeResult:
    print("\n" + "#" * 72)
    print("#  BASELINE -- Base model pi_0 : NO post-training (control)")
    print("#" * 72)
    conf = policy_pivot_confidence(ref, tok, env, device)
    print(f"  base P(rescue) at pivot = {conf:.3f}   (OOD drift = 0 by definition)")
    return ModeResult("Base pi_0 (untrained)", [], 0, 0, 0, 0.0, 0.0, conf)


def run_mode_sft(ref, tok, env, args, device) -> ModeResult:
    print("\n" + "#" * 72)
    print("#  MODE S -- SFT baseline : imitate expert actions, no KL brake")
    print("#" * 72)
    policy = fresh_policy(ref)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    sft = get_sft_trajectories()
    t0 = time.time()
    for epoch in range(args.epochs):
        print(f"\n[S][epoch {epoch + 1}/{args.epochs}]")
        for step in sft["trajectory"]:
            turn, action = step["turn"], step["sft_action"]
            prompt_text = build_prompt_text(tok, SYSTEM_PROMPT, STATE_PROMPTS[turn])
            pol_lp = action_token_logps(policy, tok, prompt_text, action, device)
            loss = -pol_lp.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            print(f"  {turn:<12} imitate={action!r:<48} CE_loss={float(loss):.4f}")
    wall = time.time() - t0
    policy.eval()
    conf = policy_pivot_confidence(policy, tok, env, device)
    kl = ood_drift(policy, ref, tok, device)
    del policy, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return ModeResult("SFT (imitation)", list(TURNS), 0, 0, args.epochs, wall, kl, conf)


def run_mode_a(ref, tok, env, args, device) -> ModeResult:
    print("\n" + "#" * 72)
    print("#  MODE A -- End-to-End GRPO : train ALL turns (terminal reward)")
    print("#" * 72)
    policy = fresh_policy(ref)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    expert = expert_actions()
    rollout_turns = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        print(f"\n[A][epoch {epoch + 1}/{args.epochs}]")
        for turn in TURNS:
            loss, st = grpo_group_loss(turn, policy, ref, tok, env, expert, args.group_size, args.beta, device)
            opt.zero_grad(); loss.backward(); opt.step()
            rollout_turns += args.group_size
            signal = "SIGNAL" if st.has_signal else "no-signal (var=0 -> zero advantage)"
            print(f"  {turn:<12} r_mean={st.mean:.3f} r_std={st.std:.3f} KL={st.kl:.4f} loss={float(loss):+.4f}  [{signal}]")
    wall = time.time() - t0
    policy.eval()
    conf = policy_pivot_confidence(policy, tok, env, device)
    kl = ood_drift(policy, ref, tok, device)
    del policy, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return ModeResult("E2E GRPO", list(TURNS), 0, rollout_turns, args.epochs, wall, kl, conf)


def run_mode_b(ref, tok, env, args, device) -> ModeResult:
    print("\n" + "#" * 72)
    print("#  MODE B -- PivotRL : DISCOVER pivots from terminal reward, then train them")
    print("#" * 72)
    policy = fresh_policy(ref)
    rng = random.Random(args.seed)

    # ---- Phase 1: discover pivots from the terminal signal ---------------- #
    pivots, report = discover_pivots(policy, tok, env, args, device, rng)
    discovery_rollouts = sum(p.n_rollouts for p in report.values())
    if not pivots:
        print("[B] no pivots discovered -- nothing to train (check --pivot-tau / --m).")

    # ---- Phase 2: train only the discovered pivot(s) ---------------------- #
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    expert = expert_actions()
    rollout_turns = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        print(f"\n[B][epoch {epoch + 1}/{args.epochs}]")
        for turn in pivots:
            loss, st = grpo_group_loss(turn, policy, ref, tok, env, expert, args.group_size, args.beta, device)
            opt.zero_grad(); loss.backward(); opt.step()
            rollout_turns += args.group_size
            print(f"  {turn:<12} r_mean={st.mean:.3f} r_std={st.std:.3f} KL={st.kl:.4f} loss={float(loss):+.4f}  [PIVOT SIGNAL]")
    wall = time.time() - t0
    policy.eval()
    conf = policy_pivot_confidence(policy, tok, env, device)
    kl = ood_drift(policy, ref, tok, device)
    del policy, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return ModeResult("PivotRL (discovered)", pivots, discovery_rollouts, rollout_turns,
                      args.epochs, wall, kl, conf, discovered_pivots=pivots)


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def _fmt_table(headers, rows) -> str:
    if _HAVE_TABULATE:
        return tabulate(rows, headers=headers, tablefmt="fancy_grid")
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    fmt = lambda cells: "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    return "\n".join([line, fmt(headers), line] + [fmt(r) for r in rows] + [line])


def print_comparison(base, s, a, b) -> None:
    turn_speedup = a.train_rollout_turns / max(b.train_rollout_turns, 1)
    drift_s = max(s.ood_kl, 1e-9)
    drift_b = max(b.ood_kl, 1e-9)
    retained_vs_sft = max(0.0, (drift_s - drift_b) / drift_s)
    base_c = base.pivot_confidence
    lift = lambda x: x.pivot_confidence - base_c

    headers = ["Metric", base.name, s.name, a.name, b.name]
    rows = [
        ["Turns trained on", "none", "all 3", "all 3", ", ".join(b.trained_turns) or "(none)"],
        ["Pivot discovery", "--", "--", "--", "from terminal reward"],
        ["Offline discovery rollouts", "0", "0", "0", str(b.discovery_rollouts)],
        ["Training rollout-turns", "0", str(s.train_rollout_turns), str(a.train_rollout_turns), str(b.train_rollout_turns)],
        ["Measured wall-clock (s)", "0.00", f"{s.wall_clock_s:.2f}", f"{a.wall_clock_s:.2f}", f"{b.wall_clock_s:.2f}"],
        ["OOD drift  KL(pi_theta||pi_0)", "0.00000", f"{s.ood_kl:.5f}", f"{a.ood_kl:.5f}", f"{b.ood_kl:.5f}"],
        ["P(rescue) at pivot", f"{base_c:.3f}", f"{s.pivot_confidence:.3f}", f"{a.pivot_confidence:.3f}", f"{b.pivot_confidence:.3f}"],
        ["  -> lift over base", "--", f"{lift(s):+.3f}", f"{lift(a):+.3f}", f"{lift(b):+.3f}"],
    ]
    print("\n")
    print("=" * 72)
    print("  TERMINAL-REWARD RESULTS  --  Base  vs.  SFT  vs.  E2E GRPO  vs.  PivotRL")
    print("=" * 72)
    print(_fmt_table(headers, rows))

    print(
        "\nInterpretation:"
        f"\n  * DISCOVERY: with reward only at the end, PivotRL had to FIND the pivot."
        f"\n    It rolled out to completion and, using action-value variance (not the"
        f"\n    naive outcome variance, which is fooled by downstream pivots),"
        f"\n    correctly isolated: {b.discovered_pivots}."
        f"\n  * COST OF DISCOVERY: that took {b.discovery_rollouts} offline completion-rollouts --"
        f"\n    the price of NOT having per-turn rewards. It is a one-time, offline cost."
        f"\n  * COMPUTE (training): PivotRL then trained only the discovered pivot"
        f"\n    ({b.train_rollout_turns} rollout-turns) vs E2E's all-turns ({a.train_rollout_turns}) -> ~{turn_speedup:.1f}x fewer."
        f"\n  * LEARNING: base P(rescue)={base_c:.3f}; PivotRL lifts it {lift(b):+.3f} to {b.pivot_confidence:.3f}."
        f"\n  * OOD: SFT drifts {s.ood_kl:.4f} from pi_0; PivotRL only {b.ood_kl:.4f}"
        f"\n    (retains ~{retained_vs_sft*100:.0f}% of the OOD capability SFT loses)."
    )


# --------------------------------------------------------------------------- #
# Preamble + entry point
# --------------------------------------------------------------------------- #
def print_preamble(args, device) -> None:
    print("=" * 72)
    print("  GOURMET CHEF SOUFFLE -- TERMINAL-REWARD PivotRL (pivot DISCOVERY)")
    print("=" * 72)
    print(
        "Reward arrives ONLY at the end (good souffle = 1, ruined = 0).\n"
        "PivotRL must DISCOVER the pivot by rolling out to completion and asking:\n"
        "  does my CHOICE at this turn change the final outcome? (action-value var)\n"
        "  -- NOT just 'is the outcome uncertain from here?' (pooled var), which is\n"
        "     fooled by downstream pivots (the credit-assignment trap).\n"
        f"\nConfig: model={args.model}  device={device}  epochs={args.epochs}  "
        f"G={args.group_size}  M={args.m}  pivot_tau={args.pivot_tau}  beta={args.beta}  lr={args.lr}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Terminal-reward PivotRL with pivot discovery.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--group-size", type=int, default=4, help="G: GRPO group size")
    p.add_argument("--m", type=int, default=16, help="M: completion rollouts per candidate during discovery")
    p.add_argument("--pivot-tau", type=float, default=0.05, help="min action-value variance to call a turn a pivot")
    p.add_argument("--beta", type=float, default=0.02)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--mode", default="all", choices=["all", "S", "A", "B"])
    return p


def main(argv: List[str] | None = None):
    args = build_arg_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print_preamble(args, device)

    env = GourmetChefTerminalEnv()
    tok, ref = load_reference(args.model, device)

    res_base = run_baseline(ref, tok, env, args, device)
    res_s = res_a = res_b = None
    if args.mode in ("all", "S"):
        res_s = run_mode_sft(ref, tok, env, args, device)
    if args.mode in ("all", "A"):
        res_a = run_mode_a(ref, tok, env, args, device)
    if args.mode in ("all", "B"):
        res_b = run_mode_b(ref, tok, env, args, device)

    if res_s and res_a and res_b:
        print_comparison(res_base, res_s, res_a, res_b)
    return res_base, res_s, res_a, res_b


if __name__ == "__main__":
    main()
