"""Generate LPO preference pairs for ONE task and ONE language.

Each example becomes a (chosen_temperature, rejected_temperature) preference
pair: evidence that, for this prompt, the chosen temperature beats the rejected
one. Those pairs are the training signal for the adaptive-temperature MLP.

Selection per example (greedy is deterministic, so it is run ONCE):

    run tau=0 once
    |
    +-- CORRECT  -> EASY case
    |     Sweep every nonzero temperature k_easy times to measure each one's
    |     success rate. chosen = tau=0 (a guaranteed 100%, and the lowest temp);
    |     rejected = the WORST nonzero temperature (lowest rate, ties -> hottest).
    |     Skip the example if every nonzero temp is also 100% (no contrast).
    |
    +-- WRONG    -> HARD case (round-robin race)
          For each round, sample every nonzero temperature ONCE. As soon as a
          round produces at least one correct answer, pick the chosen at RANDOM
          among that round's winners (equal one-shot per temp -> no ordering
          bias) and stop. rejected = tau=0, which is PROVABLY hopeless here:
          greedy is deterministic, so it fails forever on this problem.
          |
          +-- no round ever wins (k_hard rounds, all temps wrong) -> ALL-FAIL
                chosen = the hottest temperature (an injected prior: "if greedy
                is hopeless, lean fully stochastic"); rejected = tau=0. This pair
                is not measured signal, so it bypasses the contrast check.

tau=0 is NEVER both chosen and rejected: it is chosen only in the EASY case
(where rejected is drawn from nonzero temps only) and rejected only in the
HARD / ALL-FAIL cases (where chosen is nonzero).

Generation runs through THIS container's HF backend (ministral/model.py). The
task-specific parts (prompt, answer parsing, scoring) come from tasks/<task>.py.

Writes (under datasets/):
    <task>_<lang>.jsonl              the preference pairs (+ per-temperature stats)
    <task>_<lang>.readable.txt       human-readable view of every example
    <task>_<lang>.checkpoint.json    resume state
    <task>_<lang>.skipped.jsonl      examples with no usable pair

Usage:
    python generate_pairs.py --task gsm8k --lang en
    python generate_pairs.py --task gsm8k --lang en --limit 5          # sanity check
    python generate_pairs.py --task gsm8k --lang en --k-easy 5 --k-hard 20
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import time
from pathlib import Path

import model as M

HERE = Path(__file__).parent
SPLIT = "train"  # preference pairs are built from the train split

K_EASY = 5   # samples per nonzero temperature when greedy already solves it
K_HARD = 20  # max round-robin rounds when greedy fails


# ============================================================
# Per-example sweep
# ============================================================

class Sweep:
    """Accumulates samples per temperature and remembers a representative
    correct / incorrect response for each, so a pair can cite real text."""

    def __init__(self, grid: list[float]) -> None:
        self.grid = grid
        self.total = {t: 0 for t in grid}
        self.correct = {t: 0 for t in grid}
        self.truncated = {t: 0 for t in grid}
        # stored sample = (parsed_answer, full_response, was_truncated)
        self.first_correct: dict[float, tuple[str | None, str, bool]] = {}
        self.first_incorrect: dict[float, tuple[str | None, str, bool]] = {}

    def record(self, task, t: float, response: str, truncated: bool, gt) -> bool:
        parsed, _ = task.extract_answer(response)
        ok = task.is_correct(parsed, gt)
        answer = str(parsed) if parsed is not None else None
        self.total[t] += 1
        self.correct[t] += int(ok)
        if truncated:
            self.truncated[t] += 1
        bucket = self.first_correct if ok else self.first_incorrect
        bucket.setdefault(t, (answer, response, truncated))
        return ok

    def ran(self, t: float) -> bool:
        return self.total[t] > 0

    def rate(self, t: float) -> float:
        return self.correct[t] / self.total[t] if self.total[t] else 0.0

    def rate_str(self, t: float) -> str:
        return f"{self.correct[t]}/{self.total[t]}" if self.total[t] else "n/a"

    def representative(self, t: float, prefer_correct: bool) -> tuple[str | None, str | None, bool | None, bool | None]:
        """Return (answer, response, was_correct, was_truncated) for one stored
        sample at t. prefer_correct picks a correct sample first; otherwise a
        FAILED one first (so a rejected pair can show what actually went wrong)."""
        order = ((True, self.first_correct), (False, self.first_incorrect)) if prefer_correct \
            else ((False, self.first_incorrect), (True, self.first_correct))
        for was_correct, bucket in order:
            if t in bucket:
                answer, response, truncated = bucket[t]
                return answer, response, was_correct, truncated
        return None, None, None, None


# ============================================================
# IO / checkpoint
# ============================================================

def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_text(path: Path, block: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(block)


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"last_completed_index": -1, "total_pairs": 0, "total_skipped": 0,
            "easy": 0, "hard": 0, "all_fail": 0}


def save_checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def progress_str(done: int, total: int, start: float) -> str:
    elapsed = time.perf_counter() - start
    pct = 100 * done / total if total else 100.0
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    return f"{done}/{total} {pct:.0f}% | elapsed {fmt_time(elapsed)} eta {fmt_time(eta)}"


def temperature_stats(sweep: Sweep) -> list[dict]:
    rows = []
    for t in sweep.grid:
        if sweep.ran(t):
            rows.append({"temperature": t, "ran": True, "correct": sweep.correct[t],
                         "total": sweep.total[t], "truncated": sweep.truncated[t]})
        else:
            rows.append({"temperature": t, "ran": False, "correct": None,
                         "total": None, "truncated": None})
    return rows


def _indent(text: str | None, prefix: str = "    ") -> str:
    if not text:
        return prefix + "(no response captured)"
    return "\n".join(prefix + line for line in text.splitlines())


def _attempt_label(was_correct: bool | None, was_truncated: bool | None) -> str:
    if was_correct is None:
        return "no attempt captured"
    label = "a CORRECT attempt" if was_correct else "a FAILED attempt"
    if was_truncated:
        label += ", TRUNCATED at token limit"
    return label


def skip_samples(sweep: Sweep) -> list[dict]:
    """Per-temperature representative sample (answer + full response) for a
    skipped example, so a skip is auditable without re-running anything."""
    out = []
    for t in sweep.grid:
        if not sweep.ran(t):
            out.append({"temperature": t, "ran": False})
            continue
        ans, resp, ok, trunc = sweep.representative(t, prefer_correct=True)
        out.append({"temperature": t, "ran": True, "correct": sweep.correct[t],
                    "total": sweep.total[t], "parsed_answer": ans,
                    "was_correct": ok, "truncated": trunc, "response": resp})
    return out


def readable_skip_block(idx: int, total: int, question: str, gt, reason: str, sweep: Sweep) -> str:
    lines = ["=" * 70,
             f"[{idx + 1}/{total}]  {question.strip()}",
             f"correct answer: {gt}",
             f"mode: SKIPPED ({reason})",
             "temperature sweep (correct/total, [T]=#truncated):"]
    for t in sweep.grid:
        if sweep.ran(t):
            trunc = f" [T={sweep.truncated[t]}]" if sweep.truncated[t] else ""
            lines.append(f"  t={t:<4} {sweep.rate_str(t)}{trunc}")
        else:
            lines.append(f"  t={t:<4} (not run)")
    lines.append("per-temperature sample responses (one example each):")
    for t in sweep.grid:
        if not sweep.ran(t):
            continue
        ans, resp, ok, trunc = sweep.representative(t, prefer_correct=True)
        tnote = ", TRUNCATED" if trunc else ""
        lines.append(f"  --- t={t} (parsed: {ans}, {'correct' if ok else 'WRONG'}{tnote}) ---")
        lines.append(_indent(resp))
        lines.append(f"  parsed answer: {ans}")
        lines.append("")
    lines.append("")
    return "\n".join(lines)


def readable_block(idx: int, total: int, question: str, gt, mode: str,
                   sweep: Sweep, chosen_t: float, rejected_t: float,
                   chosen_ans, rejected_ans, chosen_resp, rejected_resp,
                   chosen_ok, rejected_ok, chosen_trunc, rejected_trunc) -> str:
    lines = ["=" * 70,
             f"[{idx + 1}/{total}]  {question.strip()}",
             f"correct answer: {gt}",
             f"mode: {mode}",
             "temperature sweep (correct/total, [T]=#truncated):"]
    for t in sweep.grid:
        tag = ""
        if t == chosen_t:
            tag = "   <- chosen"
        elif t == rejected_t:
            tag = "   <- rejected"
        if sweep.ran(t):
            trunc = f" [T={sweep.truncated[t]}]" if sweep.truncated[t] else ""
            rate = f"{sweep.rate_str(t)}{trunc}"
        else:
            rate = "(not run)"
        lines.append(f"  t={t:<4} {rate}{tag}")
    lines += ["preference pair:",
              f"  chosen   tau={chosen_t}  ({sweep.rate_str(chosen_t)})  parsed answer={chosen_ans}",
              f"  rejected tau={rejected_t}  ({sweep.rate_str(rejected_t)})  parsed answer={rejected_ans}",
              "",
              f"  --- chosen: complete response (tau={chosen_t}, {_attempt_label(chosen_ok, chosen_trunc)}) ---",
              _indent(chosen_resp),
              f"  parsed answer: {chosen_ans}",
              "",
              f"  --- rejected: complete response (tau={rejected_t}, {_attempt_label(rejected_ok, rejected_trunc)}) ---",
              _indent(rejected_resp),
              f"  parsed answer: {rejected_ans}",
              "", ""]
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N examples (sanity check)")
    ap.add_argument("--k-easy", type=int, default=K_EASY, help="samples per nonzero temp when greedy is correct")
    ap.add_argument("--k-hard", type=int, default=K_HARD, help="max round-robin rounds when greedy fails")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm",
                    help="vllm = batched, high-throughput (default); hf = transformers fallback (slow)")
    args = ap.parse_args()

    task = importlib.import_module(f"tasks.{args.task}")  # lazy: only the requested task
    cfg = M.load_config()

    device = M.get_device(cfg.get("device", "auto"))
    dtype = M.get_dtype(cfg["dtype_name"])
    grid = [float(t) for t in cfg["temperature_grid"]]
    if 0.0 not in grid:
        raise ValueError("temperature_grid must include 0.0 (greedy) for this scheme.")
    nonzero = [t for t in grid if t > 0.0]  # ascending
    base_seed = int(cfg["seed"])

    out_dir = HERE / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / f"{args.task}_{args.lang}.jsonl"
    ckpt_path = out_dir / f"{args.task}_{args.lang}.checkpoint.json"
    skipped_path = out_dir / f"{args.task}_{args.lang}.skipped.jsonl"
    readable_path = out_dir / f"{args.task}_{args.lang}.readable.txt"

    print(f"Task / lang   : {args.task} / {args.lang}")
    print(f"Device / dtype: {device} / {dtype}")
    print(f"Temperatures  : {grid}  (k_easy={args.k_easy}, k_hard={args.k_hard})")
    print(f"Backend       : {args.backend}")

    # gen(messages, specs) -> list[(text, truncated)] aligned 1:1 with specs,
    # where specs is a list of (temperature, seed). This is the ONLY thing the
    # two backends differ on; all pair-selection logic below is backend-agnostic.
    if args.backend == "vllm":
        llm = M.load_vllm(cfg)

        def gen(messages, specs):
            return M.vllm_generate(llm, cfg, messages, specs)
    else:
        tokenizer, mdl = M.load_model(cfg, device, dtype)

        def gen(messages, specs):
            # Fallback: one generation per spec (no cross-spec batching).
            return [M.generate_samples(mdl, tokenizer, messages, t, 1, device, cfg, s)[0]
                    for (t, s) in specs]

    examples = task.load(args.lang, SPLIT)
    if args.limit is not None:
        examples = examples[: args.limit]
    total = len(examples)

    ckpt = load_checkpoint(ckpt_path)
    start_idx = ckpt["last_completed_index"] + 1
    to_process = sum(1 for ex in examples if ex["index"] >= start_idx)
    print(f"Examples      : {total}  (starting at index {start_idx}, {to_process} to process)")
    print(f"Output        : {pairs_path}\n")

    start = time.perf_counter()
    processed = 0
    for ex in examples:
        idx = ex["index"]
        if idx < start_idx:
            continue
        processed += 1
        prog = progress_str(processed, to_process, start)
        gt = ex["ground_truth"]

        if gt is None:
            append_jsonl(skipped_path, {"index": idx, "reason": "bad_ground_truth"})
            ckpt["total_skipped"] += 1
            ckpt["last_completed_index"] = idx
            save_checkpoint(ckpt_path, ckpt)
            print(f"[{prog}] idx={idx} skipped bad ground truth")
            continue

        messages = task.build_messages(ex["question"])
        sweep = Sweep(grid)
        rng = random.Random(base_seed + idx)   # reproducible random tie-break per example
        sweep_seed = base_seed + idx * 100_000  # distinct generation seeds per example

        # --- greedy, once ---
        print(f"[{prog}] idx={idx} greedy (t=0)...", flush=True)
        (text0, trunc0), = gen(messages, [(0.0, sweep_seed)])
        sweep_seed += 1
        t0_correct = sweep.record(task, 0.0, text0, trunc0, gt)

        mode = None
        chosen_t = rejected_t = None

        if t0_correct:
            # EASY: sweep every nonzero temp k_easy times to find the worst one.
            # All nonzero x k_easy samples go out in ONE batched call; results map
            # back to their temperature in spec order. Selection is unchanged.
            specs = [(t, sweep_seed + i) for i, t in enumerate(
                t for t in nonzero for _ in range(args.k_easy))]
            sweep_seed += len(specs)
            print(f"[{prog}] idx={idx} easy sweep: {len(specs)} samples "
                  f"({len(nonzero)} temps x {args.k_easy})", flush=True)
            results = gen(messages, specs)
            for (t, _seed), (text, trunc) in zip(specs, results):
                sweep.record(task, t, text, trunc, gt)

            ran = [t for t in nonzero if sweep.ran(t)]
            worst_rate = min(sweep.rate(t) for t in ran)
            if sweep.rate(0.0) - worst_rate > 0:  # contrast exists
                mode = "easy"
                chosen_t = 0.0
                worst = [t for t in ran if sweep.rate(t) == worst_rate]
                rejected_t = rng.choice(worst)  # ties -> random (no hottest-temp bias)
            # else: every nonzero temp is also 100% -> no contrast -> skip below
        else:
            # HARD: round-robin race. Each round = one sample at every nonzero
            # temp, sent as ONE batched call. Finish the round, then pick the
            # chosen at random among that round's winners and stop. Identical
            # semantics to the per-sample loop, just batched per round.
            for _round in range(args.k_hard):
                print(f"[{prog}] idx={idx} hard race: round {_round + 1}/{args.k_hard} "
                      f"({len(nonzero)} temps)", flush=True)
                specs = [(t, sweep_seed + i) for i, t in enumerate(nonzero)]
                sweep_seed += len(specs)
                results = gen(messages, specs)
                winners = []
                for (t, _seed), (text, trunc) in zip(specs, results):
                    if sweep.record(task, t, text, trunc, gt):
                        winners.append(t)
                if winners:
                    chosen_t = rng.choice(winners)
                    rejected_t = 0.0
                    mode = "hard"
                    break
            if chosen_t is None:
                mode = "all_fail"
                chosen_t = nonzero[-1]  # hottest temperature (injected prior)
                rejected_t = 0.0

        # --- guards ---
        if chosen_t is None or rejected_t is None:
            append_jsonl(skipped_path, {
                "index": idx, "reason": "no_contrast", "question": ex["question"],
                "ground_truth": str(gt), "temperature_stats": temperature_stats(sweep),
                "samples": skip_samples(sweep),
            })
            append_text(readable_path, readable_skip_block(
                idx, total, ex["question"], gt, "no_contrast", sweep))
            ckpt["total_skipped"] += 1
            ckpt["last_completed_index"] = idx
            save_checkpoint(ckpt_path, ckpt)
            print(f"[{prog}] idx={idx} skipped (no contrast)")
            continue

        if chosen_t == rejected_t:  # must never happen; defensive
            append_jsonl(skipped_path, {
                "index": idx, "reason": "degenerate_same_temp", "question": ex["question"],
                "ground_truth": str(gt), "chosen_temperature": chosen_t,
                "temperature_stats": temperature_stats(sweep),
            })
            ckpt["total_skipped"] += 1
            ckpt["last_completed_index"] = idx
            save_checkpoint(ckpt_path, ckpt)
            print(f"[{prog}] idx={idx} skipped (degenerate pair tau={chosen_t})")
            continue

        # --- representative responses (rejected = a FAILED attempt) ---
        chosen_ans, chosen_resp, chosen_ok, chosen_trunc = sweep.representative(chosen_t, prefer_correct=True)
        rejected_ans, rejected_resp, rejected_ok, rejected_trunc = sweep.representative(rejected_t, prefer_correct=False)

        append_jsonl(pairs_path, {
            "index": idx,
            "question": ex["question"],
            "ground_truth": str(gt),
            "mode": mode,
            "k_easy": args.k_easy,
            "k_hard": args.k_hard,
            "seed": base_seed + idx * 100_000,
            "chosen": {
                "temperature": chosen_t, "rate": sweep.rate_str(chosen_t),
                "answer": chosen_ans, "response": chosen_resp,
                "was_correct": chosen_ok, "truncated": chosen_trunc,
            },
            "rejected": {
                "temperature": rejected_t, "rate": sweep.rate_str(rejected_t),
                "answer": rejected_ans, "response": rejected_resp,
                "was_correct": rejected_ok, "truncated": rejected_trunc,
            },
            "temperature_stats": temperature_stats(sweep),
        })
        append_text(readable_path, readable_block(
            idx, total, ex["question"], gt, mode, sweep,
            chosen_t, rejected_t, chosen_ans, rejected_ans,
            chosen_resp, rejected_resp, chosen_ok, rejected_ok,
            chosen_trunc, rejected_trunc))

        ckpt["total_pairs"] += 1
        ckpt[mode] = ckpt.get(mode, 0) + 1
        ckpt["last_completed_index"] = idx
        save_checkpoint(ckpt_path, ckpt)
        print(f"[{prog}] {mode}: chosen tau={chosen_t} ({sweep.rate_str(chosen_t)}) "
              f"vs rejected tau={rejected_t} ({sweep.rate_str(rejected_t)}) | pairs={ckpt['total_pairs']}")

    print(f"\nDone in {time.perf_counter() - start:.0f}s. "
          f"pairs={ckpt['total_pairs']} skipped={ckpt['total_skipped']} "
          f"(easy={ckpt.get('easy', 0)} hard={ckpt.get('hard', 0)} all_fail={ckpt.get('all_fail', 0)})")
    print(f"Pairs    -> {pairs_path}")
    print(f"Readable -> {readable_path}")


if __name__ == "__main__":
    main()
