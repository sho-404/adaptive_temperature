"""Generate LPO preference pairs for ONE task and ONE language.

For each example, samples several responses at random temperatures from the grid,
then forms a preference pair: a CORRECT response (chosen) vs an INCORRECT one
(rejected). Pairs are the training signal for the adaptive-temperature MLP.

Writes:
    datasets/<task>_<lang>.jsonl              the preference pairs
    datasets/<task>_<lang>.checkpoint.json    resume state
    datasets/<task>_<lang>.skipped.jsonl      examples with no usable pair

Generation goes through Ollama (no hidden states needed here — only responses,
their temperatures, and correctness). The task-specific parts (prompt, answer
parsing, scoring) come from tasks/<task>.py.

Usage:
    python generate_pairs.py --task gsm8k --lang en
    python generate_pairs.py --task gsm8k --lang en --limit 200
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import time
from decimal import Decimal
from pathlib import Path

import requests

HERE = Path(__file__).parent
SPLIT = "train"  # preference pairs are built from the train split

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_SHOW_URL = "http://localhost:11434/api/show"

N_RESPONSES = 16   # max generations per example
MIN_RESPONSES = 3  # min before early-stopping once a correct/incorrect contrast exists
MAX_RETRIES = 3


def load_config() -> dict:
    return json.loads((HERE / "config.json").read_text(encoding="utf-8"))


# ============================================================
# Ollama
# ============================================================

def check_model(model: str) -> None:
    try:
        requests.post(OLLAMA_SHOW_URL, json={"model": model}, timeout=30).raise_for_status()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Ollama model not available: {model}\nRun:  ollama pull {model}\n\nOriginal error: {e}"
        )


def ollama_generate(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["response"].strip()


# ============================================================
# Candidate generation / pair selection (task-agnostic)
# ============================================================

def generate_candidates(task, model, prompt, ground_truth, grid, max_tokens) -> list[dict]:
    """Sample responses at random temperatures; stop early once we have a
    correct AND an incorrect parsed answer (and at least MIN_RESPONSES)."""
    candidates: list[dict] = []
    for sample_id in range(N_RESPONSES):
        tau = random.choice(grid)
        response = ollama_generate(model, prompt, tau, max_tokens)
        parsed, status = task.extract_answer(response)
        candidates.append({
            "sample_id": sample_id, "temperature": tau, "response": response,
            "parsed_answer": str(parsed) if parsed is not None else None,
            "parse_status": status,
        })

        correct = incorrect = 0
        for c in candidates:
            if c["parsed_answer"] is None:
                continue
            if task.is_correct(Decimal(c["parsed_answer"]), ground_truth):
                correct += 1
            else:
                incorrect += 1
        if len(candidates) >= MIN_RESPONSES and correct > 0 and incorrect > 0:
            break
    return candidates


def select_pair(task, candidates, ground_truth) -> dict | None:
    correct, incorrect, unparseable = [], [], []
    for c in candidates:
        if c["parsed_answer"] is None:
            unparseable.append(c)
        elif task.is_correct(Decimal(c["parsed_answer"]), ground_truth):
            correct.append(c)
        else:
            incorrect.append(c)
    if not correct or not incorrect:
        return None
    return {
        "chosen": random.choice(correct),
        "rejected": random.choice(incorrect),
        "num_correct": len(correct),
        "num_incorrect": len(incorrect),
        "num_unparseable": len(unparseable),
    }


# ============================================================
# Checkpoint / IO
# ============================================================

def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"last_completed_index": -1, "total_pairs": 0, "total_skipped": 0}


def save_checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N examples")
    args = ap.parse_args()

    task = importlib.import_module(f"tasks.{args.task}")  # lazy: only the requested task
    cfg = load_config()
    model = cfg["ollama_model"]
    grid = [float(t) for t in cfg["temperature_grid"]]
    max_tokens = int(cfg["max_new_tokens"])

    random.seed(int(cfg["seed"]))
    check_model(model)

    out_dir = HERE / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / f"{args.task}_{args.lang}.jsonl"
    ckpt_path = out_dir / f"{args.task}_{args.lang}.checkpoint.json"
    skipped_path = out_dir / f"{args.task}_{args.lang}.skipped.jsonl"

    examples = task.load(args.lang, SPLIT)
    if args.limit is not None:
        examples = examples[: args.limit]
    total = len(examples)

    ckpt = load_checkpoint(ckpt_path)
    start_idx = ckpt["last_completed_index"] + 1

    print(f"Task / lang : {args.task} / {args.lang}")
    print(f"Ollama model: {model}")
    print(f"Examples    : {total}  (starting at index {start_idx})")
    print(f"Output      : {pairs_path}\n")

    start = time.perf_counter()
    for ex in examples:
        idx = ex["index"]
        if idx < start_idx:
            continue
        gt = ex["ground_truth"]

        if gt is None:
            append_jsonl(skipped_path, {"index": idx, "reason": "bad_ground_truth"})
            ckpt["total_skipped"] += 1
            ckpt["last_completed_index"] = idx
            save_checkpoint(ckpt_path, ckpt)
            print(f"[{idx + 1}/{total}] skipped bad ground truth")
            continue

        candidates = None
        for attempt in range(MAX_RETRIES):
            try:
                candidates = generate_candidates(task, model, task.build_prompt(ex["question"]), gt, grid, max_tokens)
                break
            except Exception as e:  # noqa: BLE001
                print(f"[{idx + 1}/{total}] generation error {attempt + 1}/{MAX_RETRIES}: {e}")
                time.sleep(3)

        if candidates is None:
            append_jsonl(skipped_path, {"index": idx, "reason": "generation_failed"})
            ckpt["total_skipped"] += 1
            ckpt["last_completed_index"] = idx
            save_checkpoint(ckpt_path, ckpt)
            print(f"[{idx + 1}/{total}] skipped generation failed")
            continue

        pair = select_pair(task, candidates, gt)
        if pair is None:
            append_jsonl(skipped_path, {
                "index": idx, "reason": "no_correct_incorrect_contrast",
                "question": ex["question"], "ground_truth": str(gt),
                "num_generated": len(candidates), "candidates": candidates,
            })
            ckpt["total_skipped"] += 1
            status = f"skipped no contrast (generated {len(candidates)})"
        else:
            append_jsonl(pairs_path, {
                "index": idx, "question": ex["question"], "ground_truth": str(gt),
                "chosen_response": pair["chosen"]["response"],
                "chosen_temperature": pair["chosen"]["temperature"],
                "chosen_answer": pair["chosen"]["parsed_answer"],
                "rejected_response": pair["rejected"]["response"],
                "rejected_temperature": pair["rejected"]["temperature"],
                "rejected_answer": pair["rejected"]["parsed_answer"],
                "num_generated": len(candidates),
                "stopped_early": len(candidates) < N_RESPONSES,
                "num_correct": pair["num_correct"], "num_incorrect": pair["num_incorrect"],
                "num_unparseable": pair["num_unparseable"],
                "all_candidates": candidates,
            })
            ckpt["total_pairs"] += 1
            status = (f"saved pair (correct={pair['num_correct']} "
                      f"incorrect={pair['num_incorrect']})")

        ckpt["last_completed_index"] = idx
        save_checkpoint(ckpt_path, ckpt)
        print(f"[{idx + 1}/{total}] {status} | pairs={ckpt['total_pairs']}")

    print(f"\nDone in {time.perf_counter() - start:.0f}s. "
          f"pairs={ckpt['total_pairs']} skipped={ckpt['total_skipped']}")
    print(f"Pairs -> {pairs_path}")


if __name__ == "__main__":
    main()
