from __future__ import annotations

import json
import os
import random
import re
import time
from datetime import datetime
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path

import requests
from datasets import load_dataset

getcontext().prec = 50

# =========================
# CONFIG
# =========================

OLLAMA_MODEL = "ministral-3:3b-instruct-2512-fp16"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_SHOW_URL = "http://localhost:11434/api/show"

DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "train"

NUM_EXAMPLES = None          # set None for full train split
N_RESPONSES = 16           # maximum generations per example
MIN_RESPONSES = 3         # minimum before early stopping can happen

TEMPERATURES = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
MAX_TOKENS = 512
SEED = 42

OUTPUT_DIR = Path("gsm8k_lpo_ollama")
OUTPUT_FILE = OUTPUT_DIR / "preference_pairs.jsonl"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"
SKIPPED_FILE = OUTPUT_DIR / "skipped_examples.jsonl"

ABS_TOL = Decimal("1e-6")

# Strong default: safer labels.
# If False, only explicit final answer markers are accepted.
ALLOW_LAST_NUMBER_FALLBACK = False

MAX_RETRIES = 3


# =========================
# NUMERIC PARSING
# =========================

def clean_number_text(text: str) -> str:
    text = text.strip()
    text = text.replace("−", "-")
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.strip()

    # If the candidate contains equality, use the right side only.
    # Example: "10 = 10.00" -> "10.00"
    if "=" in text:
        text = text.split("=")[-1]

    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[.;:]+$", "", text)
    return text


def parse_decimal(text: str) -> Decimal | None:
    """
    Parses:
      10
      10.00
      1,000
      -3.5
      3/4
      10 = 10.00
    """
    text = clean_number_text(text)

    frac = re.fullmatch(r"(-?\d+(?:\.\d+)?)\/(-?\d+(?:\.\d+)?)", text)
    if frac:
        try:
            num = Decimal(frac.group(1))
            den = Decimal(frac.group(2))
            if den == 0:
                return None
            return num / den
        except InvalidOperation:
            return None

    if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return None

    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def extract_ground_truth(answer: str) -> Decimal | None:
    match = re.search(r"####\s*([^\n]+)", answer)
    if not match:
        return None
    return parse_decimal(match.group(1))


def extract_model_answer(response: str) -> tuple[Decimal | None, str]:
    """
    Safe policy:
    - Prefer explicit final markers.
    - Accept boxed answers.
    - Do NOT use last-number fallback unless explicitly enabled.
    - Unparseable generations are excluded from preference construction.
    """

    marker_patterns = [
        r"(?m)^\s*####\s*([^\n]+?)\s*$",
        r"(?im)^\s*final answer\s*[:=]\s*([^\n]+?)\s*$",
        r"(?im)^\s*answer\s*[:=]\s*([^\n]+?)\s*$",
    ]

    for pattern in marker_patterns:
        matches = re.findall(pattern, response)
        if matches:
            value = parse_decimal(matches[-1])
            if value is not None:
                return value, "explicit_marker"

    boxed = re.findall(r"\\boxed\{([^{}]+)\}", response)
    if boxed:
        value = parse_decimal(boxed[-1])
        if value is not None:
            return value, "boxed"

    if ALLOW_LAST_NUMBER_FALLBACK:
        nums = re.findall(
            r"-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*-?\d[\d,]*(?:\.\d+)?)?",
            response,
        )
        if nums:
            value = parse_decimal(nums[-1])
            if value is not None:
                return value, "last_number_fallback"

    return None, "unparseable"


def numerically_equal(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) <= ABS_TOL


# =========================
# PROMPTING / OLLAMA
# =========================

def build_prompt(question: str) -> str:
    return f"""You are solving a grade-school math problem.

Rules:
1. Solve step by step.
2. The final line must be exactly:
#### <number>
3. The final answer must contain only the number after ####.
4. Do not include units in the final answer.
5. Do not write a sentence after the final answer.

Problem:
{question}
"""


def check_model_available() -> None:
    try:
        r = requests.post(OLLAMA_SHOW_URL, json={"model": OLLAMA_MODEL}, timeout=30)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Ollama model is not available: {OLLAMA_MODEL}\n"
            f"Run:\n  ollama pull {OLLAMA_MODEL}\n\n"
            f"Original error: {e}"
        )


def ollama_generate(prompt: str, temperature: float) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": MAX_TOKENS,
        },
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["response"].strip()


# =========================
# CHECKPOINT / IO
# =========================

def default_checkpoint() -> dict:
    return {
        "last_completed_index": -1,
        "total_processed": 0,
        "total_pairs": 0,
        "total_skipped_no_contrast": 0,
        "total_skipped_bad_ground_truth": 0,
        "total_unparseable_generations": 0,
    }


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return default_checkpoint()


def save_checkpoint(state: dict) -> None:
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, CHECKPOINT_FILE)


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


# =========================
# LPO DATASET CONSTRUCTION
# =========================

def generate_one_candidate(sample_id: int, prompt: str, temperature: float) -> dict:
    response = ollama_generate(prompt, temperature)
    parsed, parse_status = extract_model_answer(response)

    return {
        "sample_id": sample_id,
        "temperature": temperature,
        "response": response,
        "parsed_answer": str(parsed) if parsed is not None else None,
        "parse_status": parse_status,
    }


def has_correct_incorrect_contrast(
    candidates: list[dict],
    ground_truth: Decimal,
) -> tuple[bool, int, int, int]:
    correct = 0
    incorrect = 0
    unparseable = 0

    for c in candidates:
        if c["parsed_answer"] is None:
            unparseable += 1
            continue

        parsed = Decimal(c["parsed_answer"])

        if numerically_equal(parsed, ground_truth):
            correct += 1
        else:
            incorrect += 1

    has_contrast = correct > 0 and incorrect > 0
    return has_contrast, correct, incorrect, unparseable


def generate_candidates(question: str, ground_truth: Decimal) -> list[dict]:
    """
    Generates candidates sequentially.

    Stops early when:
    - at least MIN_RESPONSES have been generated, and
    - we have at least one correct and one incorrect parsed answer.

    Otherwise, it continues until N_RESPONSES.
    """

    prompt = build_prompt(question)
    candidates: list[dict] = []

    for sample_id in range(N_RESPONSES):
        tau = random.choice(TEMPERATURES)

        candidate = generate_one_candidate(
            sample_id=sample_id,
            prompt=prompt,
            temperature=tau,
        )

        candidates.append(candidate)

        has_contrast, _, _, _ = has_correct_incorrect_contrast(
            candidates,
            ground_truth,
        )

        if len(candidates) >= MIN_RESPONSES and has_contrast:
            break

    return candidates


def select_pair(candidates: list[dict], ground_truth: Decimal) -> dict | None:
    correct = []
    incorrect = []
    unparseable = []

    for c in candidates:
        if c["parsed_answer"] is None:
            unparseable.append(c)
            continue

        parsed = Decimal(c["parsed_answer"])

        if numerically_equal(parsed, ground_truth):
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


def main() -> None:
    script_start_wall = datetime.now()
    script_start_perf = time.perf_counter()

    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    check_model_available()

    print("Loading GSM8K...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)

    if NUM_EXAMPLES is not None:
        dataset = dataset.select(range(min(NUM_EXAMPLES, len(dataset))))

    total = len(dataset)
    ckpt = load_checkpoint()
    start_idx = ckpt["last_completed_index"] + 1

    print(f"Script started    : {script_start_wall.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model             : {OLLAMA_MODEL}")
    print(f"Examples          : {total}")
    print(f"Start index       : {start_idx}")
    print(f"Max responses/ex  : {N_RESPONSES}")
    print(f"Min responses/ex  : {MIN_RESPONSES}")
    print(f"Generation mode   : sequential")
    print(f"Output            : {OUTPUT_FILE}")
    print()

    recent_times = []
    processed_this_run = 0
    pairs_at_start = ckpt["total_pairs"]

    for idx in range(start_idx, total):
        row = dataset[idx]
        question = row["question"]
        ground_truth = extract_ground_truth(row["answer"])

        example_start = time.perf_counter()

        if ground_truth is None:
            ckpt["total_skipped_bad_ground_truth"] += 1
            ckpt["total_processed"] += 1
            ckpt["last_completed_index"] = idx
            save_checkpoint(ckpt)

            append_jsonl(SKIPPED_FILE, {
                "gsm8k_index": idx,
                "reason": "bad_ground_truth",
                "raw_answer": row["answer"],
            })

            processed_this_run += 1
            print(f"[{idx + 1}/{total}] skipped bad ground truth")
            continue

        candidates = None

        for attempt in range(MAX_RETRIES):
            try:
                candidates = generate_candidates(question, ground_truth)
                break
            except Exception as e:
                print(
                    f"[{idx + 1}/{total}] generation error "
                    f"attempt {attempt + 1}/{MAX_RETRIES}: {e}"
                )
                time.sleep(3)

        if candidates is None:
            append_jsonl(SKIPPED_FILE, {
                "gsm8k_index": idx,
                "reason": "generation_failed",
                "question": question,
                "ground_truth": str(ground_truth),
            })

            ckpt["total_processed"] += 1
            ckpt["last_completed_index"] = idx
            save_checkpoint(ckpt)

            processed_this_run += 1
            print(f"[{idx + 1}/{total}] skipped generation failed")
            continue

        pair = select_pair(candidates, ground_truth)
        num_unparseable = sum(c["parsed_answer"] is None for c in candidates)

        ckpt["total_unparseable_generations"] += num_unparseable

        if pair is None:
            ckpt["total_skipped_no_contrast"] += 1

            append_jsonl(SKIPPED_FILE, {
                "gsm8k_index": idx,
                "reason": "no_correct_incorrect_contrast",
                "question": question,
                "ground_truth": str(ground_truth),
                "num_generated": len(candidates),
                "num_unparseable": num_unparseable,
                "candidates": candidates,
            })

            status = (
                "skipped no contrast "
                f"| generated={len(candidates)}/{N_RESPONSES} "
                f"| unparseable={num_unparseable}/{len(candidates)}"
            )

        else:
            record = {
                "gsm8k_index": idx,
                "question": question,
                "ground_truth": str(ground_truth),

                "chosen_response": pair["chosen"]["response"],
                "chosen_temperature": pair["chosen"]["temperature"],
                "chosen_answer": pair["chosen"]["parsed_answer"],
                "chosen_parse_status": pair["chosen"]["parse_status"],

                "rejected_response": pair["rejected"]["response"],
                "rejected_temperature": pair["rejected"]["temperature"],
                "rejected_answer": pair["rejected"]["parsed_answer"],
                "rejected_parse_status": pair["rejected"]["parse_status"],

                "num_generated": len(candidates),
                "max_responses": N_RESPONSES,
                "stopped_early": len(candidates) < N_RESPONSES,

                "num_correct_of_generated": pair["num_correct"],
                "num_incorrect_of_generated": pair["num_incorrect"],
                "num_unparseable_of_generated": pair["num_unparseable"],

                "all_candidates": candidates,
            }

            append_jsonl(OUTPUT_FILE, record)
            ckpt["total_pairs"] += 1

            status = (
                "saved pair "
                f"| generated={len(candidates)}/{N_RESPONSES} "
                f"| stopped_early={len(candidates) < N_RESPONSES} "
                f"| correct={pair['num_correct']} "
                f"| incorrect={pair['num_incorrect']} "
                f"| unparseable={pair['num_unparseable']}"
            )

        ckpt["total_processed"] += 1
        ckpt["last_completed_index"] = idx
        save_checkpoint(ckpt)

        processed_this_run += 1

        dt = time.perf_counter() - example_start
        recent_times.append(dt)
        recent_times = recent_times[-25:]

        done = idx + 1
        pct = 100 * done / total
        avg = sum(recent_times) / len(recent_times)
        eta = avg * (total - done)
        elapsed = time.perf_counter() - script_start_perf

        pairs_this_run = ckpt["total_pairs"] - pairs_at_start

        print(
            f"[{done}/{total}] {pct:6.2f}% | "
            f"{status} | "
            f"pairs={ckpt['total_pairs']} | "
            f"this={dt:.1f}s | "
            f"run_elapsed={fmt_time(elapsed)} | "
            f"avg_ex={safe_rate(elapsed, processed_this_run):.2f}s | "
            f"pairs/sec={safe_rate(pairs_this_run, elapsed):.4f} | "
            f"eta={fmt_time(eta)}"
        )

    script_end_wall = datetime.now()
    total_elapsed = time.perf_counter() - script_start_perf
    pairs_this_run = ckpt["total_pairs"] - pairs_at_start

    print("\nDone.")
    print(f"Script started : {script_start_wall.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script ended   : {script_end_wall.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime  : {fmt_time(total_elapsed)}")
    print(f"Processed run  : {processed_this_run}")
    print(f"Pairs this run : {pairs_this_run}")
    print(f"Examples/sec   : {safe_rate(processed_this_run, total_elapsed):.4f}")
    print(f"Pairs/sec      : {safe_rate(pairs_this_run, total_elapsed):.4f}")
    print(f"Avg sec/example: {safe_rate(total_elapsed, processed_this_run):.2f}")
    print(f"Avg sec/pair   : {safe_rate(total_elapsed, pairs_this_run):.2f}")
    print()
    print(json.dumps(ckpt, indent=2))


if __name__ == "__main__":
    main()