"""Evaluate FIXED-temperature baselines for ONE task and ONE language.

Runs the task at each fixed temperature in config.json's grid and writes:
    results/<task>/fixed_<lang>.jsonl   raw per-example records (also used to resume)
    results/<task>/fixed_<lang>.json    summary (accuracy per temperature)

The temperature sweep, model loading, generation, resume and IO are generic;
the task-specific parts (prompt, answer parsing, scoring) come from tasks/<task>.py.
Only the requested --task is imported, so an unfinished task module can't break a run.

Model loading below is specific to THIS model container (Ministral / Mistral3).
When you copy this folder for another model, that is the part you swap.

Usage:
    python eval_fixed.py --task gsm8k --lang en
    python eval_fixed.py --task gsm8k --lang en --limit 50
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import torch
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

HERE = Path(__file__).parent
SPLIT = "test"  # fixed baselines are evaluated on the test split


# ============================================================
# Config / device
# ============================================================

def load_config() -> dict:
    return json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def get_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def get_device(requested: str) -> torch.device:
    if requested in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if requested not in ("auto", "cpu"):
        print(f"WARNING: requested {requested!r} unavailable. Falling back to CPU.")
    return torch.device("cpu")


# ============================================================
# Model (specific to this container)
# ============================================================

def load_model(cfg: dict, device: torch.device, dtype: torch.dtype):
    tokenizer = MistralCommonBackend.from_pretrained(cfg["model_name_or_path"])

    kwargs = dict(dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)
    if cfg.get("use_attention_entropy"):
        kwargs["attn_implementation"] = "eager"  # match the adaptive evaluator's path

    try:
        model = Mistral3ForConditionalGeneration.from_pretrained(cfg["model_name_or_path"], **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        model = Mistral3ForConditionalGeneration.from_pretrained(cfg["model_name_or_path"], **kwargs)

    model.to(device)
    model.eval()
    return tokenizer, model


def encode(tokenizer, messages, device, max_tokens):
    tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)
    inputs = {k: v.to(device) for k, v in tokenized.items() if hasattr(v, "to")}
    if inputs["input_ids"].shape[-1] > max_tokens:
        inputs["input_ids"] = inputs["input_ids"][:, -max_tokens:]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, -max_tokens:]
    return inputs


@torch.no_grad()
def generate(model, tokenizer, messages, temperature, device, cfg, seed):
    inputs = encode(tokenizer, messages, device, cfg["max_prompt_tokens"])
    input_len = inputs["input_ids"].shape[-1]
    do_sample = float(temperature) > 0.0

    gen_kwargs = dict(**inputs, max_new_tokens=cfg["max_new_tokens"], do_sample=do_sample, use_cache=True)
    if do_sample:
        gen_kwargs["temperature"] = max(float(temperature), 1e-5)
        gen_kwargs["top_p"] = 1.0

    torch.manual_seed(int(seed))  # seed right before generate so results are loop-order independent
    output = model.generate(**gen_kwargs)[0]
    return tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()


# ============================================================
# IO / resume
# ============================================================

def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_done(raw_path: Path) -> set[tuple[float, int]]:
    """Set of (temperature, index) pairs already in the raw file, for resuming."""
    done: set[tuple[float, int]] = set()
    if not raw_path.exists():
        return done
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "temperature" in rec and "index" in rec:
                done.add((float(rec["temperature"]), int(rec["index"])))
    return done


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N examples")
    args = ap.parse_args()

    task = importlib.import_module(f"tasks.{args.task}")  # lazy: only the requested task
    cfg = load_config()

    device = get_device(cfg.get("device", "auto"))
    dtype = get_dtype(cfg["dtype_name"])
    grid = [float(t) for t in cfg["temperature_grid"]]
    seed = int(cfg["seed"])

    out_dir = HERE / "results" / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"fixed_{args.lang}.jsonl"
    summary_path = out_dir / f"fixed_{args.lang}.json"

    print(f"Task / lang  : {args.task} / {args.lang}")
    print(f"Device / dtype: {device} / {dtype}")
    print(f"Temperatures : {grid}")

    tokenizer, model = load_model(cfg, device, dtype)
    examples = task.load(args.lang, SPLIT)
    if args.limit is not None:
        examples = examples[: args.limit]
    total = len(examples)

    done = load_done(raw_path)
    # counts[temp] -> dict of tallies
    counts = {t: {"correct": 0, "incorrect": 0, "unparseable": 0, "bad_gt": 0, "failed": 0} for t in grid}
    # rebuild counts from any resumed records
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = float(rec.get("temperature", -1))
                if t not in counts:
                    continue
                if rec.get("bad_ground_truth"):
                    counts[t]["bad_gt"] += 1
                elif rec.get("generation_failed"):
                    counts[t]["failed"] += 1
                elif rec.get("parsed_answer") is None:
                    counts[t]["unparseable"] += 1
                elif rec.get("is_correct"):
                    counts[t]["correct"] += 1
                else:
                    counts[t]["incorrect"] += 1

    print(f"Examples     : {total}  (resuming {len(done)} already-done runs)\n")
    start = time.perf_counter()

    for t in grid:
        print(f"--- temperature = {t} ---")
        for ex in examples:
            idx = ex["index"]
            if (t, idx) in done:
                continue

            gt = ex["ground_truth"]
            if gt is None:
                append_jsonl(raw_path, {"temperature": t, "index": idx, "bad_ground_truth": True})
                counts[t]["bad_gt"] += 1
                continue

            try:
                response = generate(
                    model, tokenizer, task.build_messages(ex["question"]),
                    t, device, cfg, seed=seed + idx,
                )
            except Exception as e:  # noqa: BLE001 — log and record as failed, keep going
                print(f"  [T={t} idx={idx}] generation error: {e}")
                append_jsonl(raw_path, {
                    "temperature": t, "index": idx, "generation_failed": True,
                    "parsed_answer": None, "is_correct": False,
                })
                counts[t]["failed"] += 1
                continue

            parsed, status = task.extract_answer(response)
            correct = task.is_correct(parsed, gt)
            if parsed is None:
                counts[t]["unparseable"] += 1
            elif correct:
                counts[t]["correct"] += 1
            else:
                counts[t]["incorrect"] += 1

            append_jsonl(raw_path, {
                "temperature": t, "index": idx, "question": ex["question"],
                "ground_truth": str(gt), "response": response,
                "parsed_answer": str(parsed) if parsed is not None else None,
                "parse_status": status, "is_correct": correct,
            })

        c = counts[t]
        answered = c["correct"] + c["incorrect"] + c["unparseable"] + c["failed"]
        acc = c["correct"] / answered if answered else 0.0
        print(f"  >> T={t}: accuracy={acc:.4f} ({c['correct']}/{answered})\n")

    # summary
    results = []
    for t in grid:
        c = counts[t]
        answered = c["correct"] + c["incorrect"] + c["unparseable"] + c["failed"]
        results.append({
            "temperature": t,
            "accuracy": (c["correct"] / answered if answered else 0.0),
            "correct": c["correct"], "incorrect": c["incorrect"],
            "unparseable": c["unparseable"], "generation_failed": c["failed"],
            "bad_ground_truth": c["bad_gt"], "answered": answered,
        })

    summary = {
        "task": args.task, "lang": args.lang, "method": "fixed",
        "model": cfg["model_name_or_path"], "split": SPLIT,
        "num_examples": total, "seed": seed,
        "temperatures": grid, "device": str(device),
        "total_runtime_seconds": time.perf_counter() - start,
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Raw     -> {raw_path}")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
