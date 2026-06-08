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
        self.first_correct: dict[float, tuple[str | None, str]] = {}
        self.first_incorrect: dict[float, tuple[str | None, str]] = {}

    def record(self, task, t: float, response: str, gt) -> bool:
        parsed, _ = task.extract_answer(response)
        ok = task.is_correct(parsed, gt)
        answer = str(parsed) if parsed is not None else None
        self.total[t] += 1
        self.correct[t] += int(ok)
        bucket = self.first_correct if ok else self.first_incorrect
        bucket.setdefault(t, (answer, response))
        return ok

    def ran(self, t: float) -> bool:
        return self.total[t] > 0

    def rate(self, t: float) -> float:
        return self.correct[t] / self.total[t] if self.total[t] else 0.0

    def rate_str(self, t: float) -> str:
        return f"{self.correct[t]}/{self.total[t]}" if self.total[t] else "n/a"

    def representative(self, t: float, prefer_correct: bool) -> tuple[str | None, str | None, bool | None]:
        """Return (answer, response, was_correct) for one stored sample at t.
        prefer_correct picks a correct sample first; otherwise a FAILED one
        first (so a rejected pair can show what actually went wrong)."""
        order = ((True, self.first_correct), (False, self.first_incorrect)) if prefer_correct \
            else ((False, self.first_incorrect), (True, self.first_correct))
        for was_correct, bucket in order:
            if t in bucket:
                answer, response = bucket[t]
                return answer, response, was_correct
        return None, None, None


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
            rows.append({"temperature": t, "ran": True,
                         "correct": sweep.correct[t], "total": sweep.total[t]})
        else:
            rows.append({"temperature": t, "ran": False, "correct": None, "total": None})
    return rows


def _indent(text: str | None, prefix: str = "    ") -> str:
    if not text:
        return prefix + "(no response captured)"
    return "\n".join(prefix + line for line in text.splitlines())


def _attempt_label(was_correct: bool | None) -> str:
    if was_correct is None:
        return "no attempt captured"
    return "a CORRECT attempt" if was_correct else "a FAILED attempt"


def readable_block(idx: int, total: int, question: str, gt, mode: str,
                   sweep: Sweep, chosen_t: float, rejected_t: float,
                   chosen_ans, rejected_ans, chosen_resp, rejected_resp,
                   chosen_ok, rejected_ok) -> str:
    lines = ["=" * 70,
             f"[{idx + 1}/{total}]  {question.strip()}",
             f"correct answer: {gt}",
             f"mode: {mode}",
             "temperature sweep:"]
    for t in sweep.grid:
        tag = ""
        if t == chosen_t:
            tag = "   <- chosen"
        elif t == rejected_t:
            tag = "   <- rejected"
        rate = sweep.rate_str(t) if sweep.ran(t) else "(not run)"
        lines.append(f"  t={t:<4} {rate}{tag}")
    lines += ["preference pair:",
              f"  chosen   tau={chosen_t}  ({sweep.rate_str(chosen_t)})  answer={chosen_ans}",
              f"  rejected tau={rejected_t}  ({sweep.rate_str(rejected_t)})  answer={rejected_ans}",
              "",
              f"  --- chosen response  (tau={chosen_t}, {_attempt_label(chosen_ok)}) ---",
              _indent(chosen_resp),
              "",
              f"  --- rejected response (tau={rejected_t}, {_attempt_label(rejected_ok)}) ---",
              _indent(rejected_resp),
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

    tokenizer, mdl = M.load_model(cfg, device, dtype)
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
        r0 = M.generate_samples(mdl, tokenizer, messages, 0.0, 1, device, cfg, sweep_seed)
        sweep_seed += 1
        t0_correct = sweep.record(task, 0.0, r0[0], gt)

        mode = None
        chosen_t = rejected_t = None

        if t0_correct:
            # EASY: full sweep of nonzero temps to find the worst one.
            for t in nonzero:
                resps = M.generate_samples(mdl, tokenizer, messages, t, args.k_easy, device, cfg, sweep_seed)
                sweep_seed += 1
                for r in resps:
                    sweep.record(task, t, r, gt)

            ran = [t for t in nonzero if sweep.ran(t)]
            worst_rate = min(sweep.rate(t) for t in ran)
            if sweep.rate(0.0) - worst_rate > 0:  # contrast exists
                mode = "easy"
                chosen_t = 0.0
                worst = [t for t in ran if sweep.rate(t) == worst_rate]
                rejected_t = rng.choice(worst)  # ties -> random (no hottest-temp bias)
            # else: every nonzero temp is also 100% -> no contrast -> skip below
        else:
            # HARD: round-robin race. Finish each round, pick chosen at random
            # among that round's winners.
            for _round in range(args.k_hard):
                winners = []
                for t in nonzero:
                    r = M.generate_samples(mdl, tokenizer, messages, t, 1, device, cfg, sweep_seed)
                    sweep_seed += 1
                    if sweep.record(task, t, r[0], gt):
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
            })
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
        chosen_ans, chosen_resp, chosen_ok = sweep.representative(chosen_t, prefer_correct=True)
        rejected_ans, rejected_resp, rejected_ok = sweep.representative(rejected_t, prefer_correct=False)

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
                "answer": chosen_ans, "response": chosen_resp, "was_correct": chosen_ok,
            },
            "rejected": {
                "temperature": rejected_t, "rate": sweep.rate_str(rejected_t),
                "answer": rejected_ans, "response": rejected_resp, "was_correct": rejected_ok,
            },
            "temperature_stats": temperature_stats(sweep),
        })
        append_text(readable_path, readable_block(
            idx, total, ex["question"], gt, mode, sweep,
            chosen_t, rejected_t, chosen_ans, rejected_ans,
            chosen_resp, rejected_resp, chosen_ok, rejected_ok))

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
