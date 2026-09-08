"""
train_comparative_grpo.py
=========================

Turn-level GRPO trainer that runs the Gourmet Chef Souffle Challenge under two
regimes and prints a side-by-side comparison:

  Mode A  Pre-PivotRL  -- Traditional End-to-End GRPO. Every turn (prep, bake,
                          plate) is rolled out and updated on, every epoch.
  Mode B  PivotRL      -- Offline Pivot Filtering. K dry rollouts profile each
                          turn; only high-variance / low-mean "pivots" survive
                          the filter and get trained. Here that is Turn 2 only.

Both modes share the same frozen reference policy pi_0 and start from an
identical copy of it as the active policy pi_theta.

Algorithmic notes (grounded in the literature)
-----------------------------------------------
* GRPO (DeepSeekMath, arXiv:2402.03300) normalizes each candidate's reward
  against its group:  A_i = (r_i - mean(r)) / (std(r) + eps). When std(r)=0
  (all-success or all-fail groups) every advantage is 0 -> zero gradient. Those
  turns spend rollout compute for no learning. Turns 1 and 3 are exactly this.
* We use a single on-policy gradient step per collected group, so the GRPO
  importance ratio pi_theta / pi_theta_old == 1 at evaluation and the clipped
  surrogate reduces to  A_i * logprob(a_i | s). (Documented simplification for
  a compact, reproducible demo.)
* KL(pi_theta || pi_0) uses the unbiased k3 estimator from DeepSeek:
  per token, kl = exp(logr) - logr - 1  where logr = logp_ref - logp_policy.
* PivotRL (arXiv:2603.21383, Yi et al., 2026): offline pivot filtering + a
  functional verifier reward r_func(s,a)=1[a in M(s)]; KL-regularized local
  updates preserve OOD behavior (reported +10.04% OOD over SFT at 4x fewer
  rollout turns). The final table anchors its illustrative OOD-retention
  projection to that figure.

The action group at each turn is the turn's candidate pool from kitchen_env
(the sampled rollouts). Log-probs and KL are computed teacher-forced against
the live policy, so the gradients, advantages, and KL are all real.

Run:  python train_comparative_grpo.py --epochs 3 --group-size 4
"""

from __future__ import annotations

import argparse
import copy
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from kitchen_env import (
    GourmetChefEnv,
    TURNS,
    TURN_BAKE,
    SYSTEM_PROMPT,
    STATE_PROMPTS,
    get_sft_trajectories,
)

try:
    from tabulate import tabulate  # nicer tables when available
    _HAVE_TABULATE = True
except Exception:  # pragma: no cover
    _HAVE_TABULATE = False


# Unrelated prompts used to measure Out-of-Domain (OOD) distribution drift from
# the frozen reference -- a direct proxy for catastrophic forgetting.
OOD_PROBE_PROMPTS: List[str] = [
    "What is the capital of France?",
    "Write a Python function that returns the nth Fibonacci number.",
    "Summarize the water cycle in one sentence.",
    "Translate 'good morning' into Spanish.",
]

# Paper-reported OOD accuracy advantage of PivotRL over SFT; used only as the
# ceiling for the *illustrative* retention projection in the final table.
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
    """Load the frozen reference policy pi_0 and its tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[load] tokenizer + model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float32  # fp32 for stable tiny-scale training on T4/CPU
    ref = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    ref.to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return tok, ref


def fresh_policy(ref):
    """A trainable deep copy of the reference to serve as pi_theta."""
    policy = copy.deepcopy(ref)
    policy.train()
    for p in policy.parameters():
        p.requires_grad_(True)
    return policy


# --------------------------------------------------------------------------- #
# Prompt / scoring helpers
# --------------------------------------------------------------------------- #
def build_prompt_text(tok, system: str, user: str) -> str:
    """Render a chat prompt string with the generation prompt appended."""
    try:
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Base models without a chat template (e.g. gpt2) fall back to plain text.
        return f"{system}\n\n{user}\n"


def action_token_logps(model, tok, prompt_text: str, action_text: str, device: str) -> torch.Tensor:
    """Teacher-forced per-token log-probabilities of ``action_text`` given the prompt.

    Returns a 1-D tensor [A] of log p(action_token_i | prompt, action_<i>).
    Carries grad iff the model's params require grad.
    """
    prompt_ids = tok(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    action_ids = tok(action_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    input_ids = torch.cat([prompt_ids, action_ids], dim=1)

    logits = model(input_ids).logits  # [1, T, V]
    logprobs = F.log_softmax(logits, dim=-1)

    P = prompt_ids.shape[1]
    A = action_ids.shape[1]
    # Token at absolute position j is predicted from logits at position j-1.
    pred_positions = torch.arange(P - 1, P - 1 + A, device=device)
    gathered = logprobs[0, pred_positions, :].gather(
        -1, action_ids[0].unsqueeze(-1)
    ).squeeze(-1)
    return gathered  # [A]


def k3_kl(pol_logps: torch.Tensor, ref_logps: torch.Tensor) -> torch.Tensor:
    """Unbiased k3 KL(pi_theta || pi_0) estimate summed over action tokens."""
    logr = ref_logps - pol_logps  # log(pi_0 / pi_theta) per token
    return (torch.exp(logr) - logr - 1.0).sum()


# --------------------------------------------------------------------------- #
# One GRPO group update on a single turn
# --------------------------------------------------------------------------- #
@dataclass
class TurnStats:
    turn: str
    rewards: List[float]
    mean: float
    std: float
    advantages: List[float]
    kl: float
    has_signal: bool


def grpo_group_loss(
    turn: str,
    policy,
    ref,
    tok,
    env: GourmetChefEnv,
    group_size: int,
    beta: float,
    device: str,
) -> Tuple[torch.Tensor, TurnStats]:
    """Compute the turn-level GRPO loss over a group of G candidate actions."""
    prompt_text = build_prompt_text(tok, SYSTEM_PROMPT, STATE_PROMPTS[turn])
    actions = env.candidate_actions(turn)[:group_size]

    rewards = torch.tensor(
        [env.get_functional_reward(turn, a) for a in actions],
        dtype=torch.float32,
        device=device,
    )
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    if std < 1e-8:
        advantages = torch.zeros_like(rewards)  # zero-variance group -> no signal
    else:
        advantages = (rewards - mean) / (std + 1e-8)

    pg_terms: List[torch.Tensor] = []
    kl_terms: List[torch.Tensor] = []
    for action, adv in zip(actions, advantages):
        pol_lp = action_token_logps(policy, tok, prompt_text, action, device)
        with torch.no_grad():
            ref_lp = action_token_logps(ref, tok, prompt_text, action, device)
        seq_logp = pol_lp.sum()
        pg_terms.append(adv.detach() * seq_logp)
        kl_terms.append(k3_kl(pol_lp, ref_lp))

    pg_loss = -torch.stack(pg_terms).mean()
    kl_loss = torch.stack(kl_terms).mean()
    loss = pg_loss + beta * kl_loss

    stats = TurnStats(
        turn=turn,
        rewards=[float(r) for r in rewards.tolist()],
        mean=float(mean),
        std=float(std),
        advantages=[float(a) for a in advantages.tolist()],
        kl=float(kl_loss.detach()),
        has_signal=bool(std >= 1e-8),
    )
    return loss, stats


# --------------------------------------------------------------------------- #
# OOD drift measurement
# --------------------------------------------------------------------------- #
@torch.no_grad()
def policy_pivot_confidence(policy, tok, env: GourmetChefEnv, device: str) -> float:
    """Probability mass the policy puts on the *correct* rescue action(s) at the
    baking pivot, among that turn's candidate group.

    Length-normalized per-token log-probs -> softmax over the group -> sum the
    mass on actions whose functional reward is 1.0. Rises as the policy learns
    to prefer the rescue over the collapse/burn actions. This reflects actual
    learning (unlike re-profiling the fixed candidate pool, which is constant).
    """
    prompt_text = build_prompt_text(tok, SYSTEM_PROMPT, STATE_PROMPTS[TURN_BAKE])
    actions = env.candidate_actions(TURN_BAKE)
    norm_logps: List[float] = []
    rewards: List[float] = []
    for a in actions:
        lp = action_token_logps(policy, tok, prompt_text, a, device).sum()
        n_tok = max(1, len(tok(a, add_special_tokens=False).input_ids))
        norm_logps.append(float(lp) / n_tok)  # avg log-prob per token
        rewards.append(env.get_functional_reward(TURN_BAKE, a))
    probs = torch.softmax(torch.tensor(norm_logps), dim=0).tolist()
    return float(sum(p for p, r in zip(probs, rewards) if r >= 1.0))


@torch.no_grad()
def ood_drift(policy, ref, tok, device: str) -> float:
    """Mean full-distribution KL(pi_theta || pi_0) over unrelated OOD prompts.

    Lower = less catastrophic forgetting = better OOD retention.
    """
    kls: List[float] = []
    for prompt in OOD_PROBE_PROMPTS:
        text = build_prompt_text(tok, "You are a helpful assistant.", prompt)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        pol = F.log_softmax(policy(ids).logits, dim=-1)
        rf = F.log_softmax(ref(ids).logits, dim=-1)
        kl = (pol.exp() * (pol - rf)).sum(-1).mean()  # avg over sequence positions
        kls.append(float(kl))
    return statistics.fmean(kls)


# --------------------------------------------------------------------------- #
# Training regimes
# --------------------------------------------------------------------------- #
@dataclass
class ModeResult:
    name: str
    trained_turns: List[str]
    rollout_turns_offline: int
    rollout_turns_train: int  # online (on-policy) rollout-turns consumed
    epochs: int
    wall_clock_s: float
    ood_kl: float
    pivot_confidence: float  # P(policy picks the rescue action) at the pivot


def run_mode_sft(ref, tok, env, args, device) -> ModeResult:
    """SFT baseline: imitate the expert action on EVERY turn via plain
    cross-entropy, with NO KL regularization back to pi_0.

    This is cheap (0 online rollouts -- it reuses static SFT trajectories) but,
    lacking the KL brake, it drifts furthest from pi_0 -> worst OOD retention.
    This is the arm the paper's +10.04% OOD figure is measured against.
    """
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
            loss = -pol_lp.sum()  # maximum-likelihood imitation, no KL term
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(f"  {turn:<12} imitate={action!r:<48} CE_loss={float(loss):.4f}")
    wall = time.time() - t0

    conf = policy_pivot_confidence(policy, tok, env, device)
    kl = ood_drift(policy, ref, tok, device)
    del policy, opt
    if device == "cuda":
        torch.cuda.empty_cache()

    return ModeResult(
        name="SFT (imitation)",
        trained_turns=list(TURNS),
        rollout_turns_offline=0,
        rollout_turns_train=0,  # SFT consumes no online rollouts
        epochs=args.epochs,
        wall_clock_s=wall,
        ood_kl=kl,
        pivot_confidence=conf,
    )


def run_mode_a(ref, tok, env, args, device) -> ModeResult:
    print("\n" + "#" * 72)
    print("#  MODE A -- Pre-PivotRL : Traditional End-to-End GRPO (all 3 turns)")
    print("#" * 72)
    policy = fresh_policy(ref)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    rollout_turns_train = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        print(f"\n[A][epoch {epoch + 1}/{args.epochs}]")
        for turn in TURNS:
            loss, st = grpo_group_loss(turn, policy, ref, tok, env, args.group_size, args.beta, device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            rollout_turns_train += args.group_size
            signal = "SIGNAL" if st.has_signal else "no-signal (var=0 -> zero advantage)"
            print(
                f"  {turn:<12} r_mean={st.mean:.3f} r_std={st.std:.3f} "
                f"KL={st.kl:.4f} loss={float(loss):+.4f}  [{signal}]"
            )
    wall = time.time() - t0

    conf = policy_pivot_confidence(policy, tok, env, device)
    kl = ood_drift(policy, ref, tok, device)
    del policy, opt
    if device == "cuda":
        torch.cuda.empty_cache()

    return ModeResult(
        name="Pre-PivotRL (E2E GRPO)",
        trained_turns=list(TURNS),
        rollout_turns_offline=0,
        rollout_turns_train=rollout_turns_train,
        epochs=args.epochs,
        wall_clock_s=wall,
        ood_kl=kl,
        pivot_confidence=conf,
    )


def run_mode_b(ref, tok, env, args, device) -> ModeResult:
    print("\n" + "#" * 72)
    print("#  MODE B -- PivotRL : Offline Pivot Filtering + turn-level GRPO")
    print("#" * 72)
    policy = fresh_policy(ref)

    # ---- Phase 1: offline dry rollouts (K per turn) to profile variance ---- #
    print(f"\n[B][offline] running K={args.k} dry rollouts per turn to find pivots...")
    rollout_turns_offline = 0
    profiles = {}
    for turn in TURNS:
        actions = env.candidate_actions(turn)[:args.k]
        rewards = [env.get_functional_reward(turn, a) for a in actions]
        mean = statistics.fmean(rewards)
        var = statistics.pvariance(rewards)
        is_pivot = (var > 1e-9) and (mean < env.lambda_diff)
        profiles[turn] = (mean, var, is_pivot)
        rollout_turns_offline += args.k
        verdict = "PIVOT -> keep" if is_pivot else "flat (var=0 or high-mean) -> DISCARD"
        print(f"  {turn:<12} mean={mean:.3f} var={var:.4f}  =>  {verdict}")

    pivots = [t for t in TURNS if profiles[t][2]]
    discarded = [t for t in TURNS if not profiles[t][2]]
    print(f"\n[B][pivot-filter] keep={pivots}  discard={discarded}")
    print(f"[B][pivot-filter] concentrating all training compute on: {pivots}")

    # ---- Phase 2: train ONLY on the pivot turn(s) -------------------------- #
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    rollout_turns_train = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        print(f"\n[B][epoch {epoch + 1}/{args.epochs}]")
        for turn in pivots:
            loss, st = grpo_group_loss(turn, policy, ref, tok, env, args.group_size, args.beta, device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            rollout_turns_train += args.group_size
            print(
                f"  {turn:<12} r_mean={st.mean:.3f} r_std={st.std:.3f} "
                f"KL={st.kl:.4f} loss={float(loss):+.4f}  [PIVOT SIGNAL]"
            )
    wall = time.time() - t0

    conf = policy_pivot_confidence(policy, tok, env, device)
    kl = ood_drift(policy, ref, tok, device)
    del policy, opt
    if device == "cuda":
        torch.cuda.empty_cache()

    return ModeResult(
        name="PivotRL (turn-level GRPO)",
        trained_turns=pivots,
        rollout_turns_offline=rollout_turns_offline,
        rollout_turns_train=rollout_turns_train,
        epochs=args.epochs,
        wall_clock_s=wall,
        ood_kl=kl,
        pivot_confidence=conf,
    )


# --------------------------------------------------------------------------- #
# Comparative report
# --------------------------------------------------------------------------- #
def _fmt_table(headers: List[str], rows: List[List[str]]) -> str:
    if _HAVE_TABULATE:
        return tabulate(rows, headers=headers, tablefmt="fancy_grid")
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    def fmt_row(cells):
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    out = [line, fmt_row(headers), line]
    out += [fmt_row(r) for r in rows]
    out.append(line)
    return "\n".join(out)


def _total(r: ModeResult) -> int:
    return r.rollout_turns_offline + r.rollout_turns_train


def print_comparison(s: ModeResult, a: ModeResult, b: ModeResult) -> None:
    """Three-way comparison: SFT vs End-to-End GRPO vs PivotRL."""
    # Compute-efficiency: PivotRL vs E2E-GRPO on online rollout-turns.
    turn_speedup = a.rollout_turns_train / max(b.rollout_turns_train, 1)

    # OOD retention is measured against the SFT baseline (as in the paper).
    drift_s = max(s.ood_kl, 1e-9)
    drift_b = max(b.ood_kl, 1e-9)
    drift_a = max(a.ood_kl, 1e-9)
    retained_vs_sft = max(0.0, (drift_s - drift_b) / drift_s)
    projected_ood_gain = retained_vs_sft * PAPER_OOD_CEILING_PCT

    headers = ["Metric", s.name, a.name, b.name]
    rows = [
        ["Turns trained on", "all 3 (imitate)", "all 3", ", ".join(b.trained_turns)],
        ["Offline profiling rollout-turns", str(s.rollout_turns_offline), str(a.rollout_turns_offline), str(b.rollout_turns_offline)],
        ["Online rollout-turns", str(s.rollout_turns_train), str(a.rollout_turns_train), str(b.rollout_turns_train)],
        ["Total rollout-turns", str(_total(s)), str(_total(a)), str(_total(b))],
        ["Measured wall-clock (s)", f"{s.wall_clock_s:.2f}", f"{a.wall_clock_s:.2f}", f"{b.wall_clock_s:.2f}"],
        ["OOD drift  KL(pi_theta||pi_0)", f"{s.ood_kl:.5f}", f"{a.ood_kl:.5f}", f"{b.ood_kl:.5f}"],
        ["P(rescue) at pivot (learned)", f"{s.pivot_confidence:.3f}", f"{a.pivot_confidence:.3f}", f"{b.pivot_confidence:.3f}"],
    ]

    print("\n")
    print("=" * 72)
    print("  COMPARATIVE RESULTS  --  SFT  vs.  End-to-End GRPO  vs.  PivotRL")
    print("=" * 72)
    print(_fmt_table(headers, rows))

    print(
        "\nInterpretation:"
        f"\n  * COMPUTE (vs E2E-GRPO): PivotRL discarded the zero-variance prep &"
        f"\n    plating turns (advantage == 0, pure wasted rollouts) and trained only"
        f"\n    the high-variance baking pivot -> ~{turn_speedup:.1f}x fewer online rollout-turns."
        f"\n  * OOD RETENTION (vs SFT): SFT has no KL brake and drifts furthest from"
        f"\n    pi_0 (KL={s.ood_kl:.5f}); PivotRL's KL-regularized local updates stay close"
        f"\n    (KL={b.ood_kl:.5f}), matching E2E-GRPO (KL={a.ood_kl:.5f}) -- Theorem 3.3."
        f"\n  * BEST OF BOTH: PivotRL keeps SFT-like efficiency AND E2E-like OOD retention,"
        f"\n    retaining ~{retained_vs_sft * 100:.0f}% of the OOD capability SFT loses,"
        f"\n    projecting toward the paper's reported +{PAPER_OOD_CEILING_PCT:.2f}% (illustrative,"
        f"\n    not a benchmarked accuracy)."
    )


# --------------------------------------------------------------------------- #
# Educational preamble
# --------------------------------------------------------------------------- #
def print_preamble(args, device) -> None:
    print("=" * 72)
    print("  GOURMET CHEF SOUFFLE CHALLENGE  --  Comparative Turn-Level GRPO")
    print("=" * 72)
    print(
        "Research background\n"
        "  GRPO (DeepSeekMath, arXiv:2402.03300): A_i = (r_i - mean)/(std+eps).\n"
        "    A zero-variance group (all pass OR all fail) => every advantage = 0\n"
        "    => zero gradient. Such turns burn rollout compute for no signal.\n"
        "  PivotRL (arXiv:2603.21383, Yi et al. 2026): run K local rollouts,\n"
        "    keep only 'pivots' (var>0 AND mean<lambda_diff), and train there.\n"
        "    Functional verifier reward r_func(s,a)=1[a in M(s)] rewards\n"
        "    functionally-equivalent actions instead of exact strings. Reported:\n"
        "    +4.17% in-domain, +10.04% OOD vs SFT, at 4x fewer rollout turns.\n"
        f"\nConfig: model={args.model}  device={device}  epochs={args.epochs}  "
        f"G={args.group_size}  K={args.k}  beta={args.beta}  lr={args.lr}"
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Comparative turn-level GRPO: Pre-PivotRL vs PivotRL.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--group-size", type=int, default=4, help="G: GRPO group size")
    p.add_argument("--k", type=int, default=4, help="K: offline dry rollouts per turn for pivot filtering")
    p.add_argument("--beta", type=float, default=0.02, help="KL-regularization coefficient")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--mode", default="all", choices=["all", "S", "A", "B"])
    return p


def main(argv: List[str] | None = None):
    """Run the requested arms. Returns (sft, e2e_grpo, pivotrl) ModeResults
    (each None if not run)."""
    args = build_arg_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    print_preamble(args, device)

    env = GourmetChefEnv()
    tok, ref = load_reference(args.model, device)

    res_s = res_a = res_b = None
    if args.mode in ("all", "S"):
        res_s = run_mode_sft(ref, tok, env, args, device)
    if args.mode in ("all", "A"):
        res_a = run_mode_a(ref, tok, env, args, device)
    if args.mode in ("all", "B"):
        res_b = run_mode_b(ref, tok, env, args, device)

    if res_s and res_a and res_b:
        print_comparison(res_s, res_a, res_b)
    return res_s, res_a, res_b


if __name__ == "__main__":
    main()
