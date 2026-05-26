from __future__ import annotations

import json
import re
import time
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

getcontext().prec = 50

# ============================================================
# CONFIG - edit here, not through CLI args
# ============================================================

MODEL_NAME_OR_PATH = "mistralai/Ministral-3-3B-Instruct-2512-BF16"

# Read the adaptive checkpoint config so classic uses the same prompt/token/grid settings.
CKPT_DIR = Path("adaptive_decoder_ckpt")
CONFIG_FILE = CKPT_DIR / "config.json"

DEVICE = "mps"
DTYPE_NAME = "float16"

DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "test"
NUM_EXAMPLES = None  # None = full test set

OUTPUT_DIR = Path("gsm8k_eval_classic_hf")
RESULTS_FILE = OUTPUT_DIR / "results.jsonl"
SUMMARY_TXT_FILE = OUTPUT_DIR / "summary.txt"
SUMMARY_JSON_FILE = OUTPUT_DIR / "summary.json"

# Alter this array freely only if CLASSIC_TEMPERATURES_OVERRIDE is not None.
TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Strict comparability mode:
# - If True, classic reads adaptive_decoder_ckpt/config.json and uses the same
#   temperature grid and max_prompt_tokens that the adaptive evaluator uses.
# - Set CLASSIC_TEMPERATURES_OVERRIDE to a list if you intentionally want a different grid.
USE_CKPT_CONFIG_FOR_COMPARABILITY = True
CLASSIC_TEMPERATURES_OVERRIDE = None  # example: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

MAX_PROMPT_TOKENS = 768
MAX_NEW_TOKENS = 512
SEED = 42
MAX_RETRIES = 2
ABS_TOL = Decimal("1e-6")
ALLOW_LAST_NUMBER_FALLBACK = False

# Strict comparability: match adaptive model-loading path.
# Adaptive uses eager attention when attention entropy is enabled.
# Classic does not request attentions during generation, but loading with the same
# attention implementation removes an avoidable runtime mismatch.
USE_ATTENTION_ENTROPY = True


# ============================================================
# Numeric parsing
# ============================================================

def clean_number_text(text: str) -> str:
    text = text.strip().replace("−", "-").replace(",", "").replace("$", "").strip()
    if "=" in text:
        text = text.split("=")[-1]
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[.;:]+$", "", text)
    return text


def parse_decimal(text: str) -> Decimal | None:
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


def seed_generation(seed: int | None) -> None:
    """Seed immediately before generate() so results do not depend on loop order."""
    if seed is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# ============================================================
# Prompt / tokenizer helpers
# Same as adaptive evaluator
# ============================================================

def make_messages(question: str) -> list[dict[str, str]]:
    system = (
        "You are a careful grade-school math solver. "
        "You must obey the requested final-answer format exactly."
    )

    user = f"""Solve the following grade-school math problem step by step.

Rules:
1. The final line must be exactly: #### <number>
2. The final answer must contain only the number after ####.
3. Do not include units in the final answer.
4. Do not write anything after the final #### line.

Problem:
{question}"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def encode_messages(
    tokenizer,
    messages: list[dict[str, str]],
    device: torch.device,
    max_tokens: int,
) -> dict[str, torch.Tensor]:
    tokenized = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
    )

    inputs: dict[str, torch.Tensor] = {}
    for k, v in tokenized.items():
        if hasattr(v, "to"):
            inputs[k] = v.to(device)

    seq_len = inputs["input_ids"].shape[-1]

    if seq_len > max_tokens:
        inputs["input_ids"] = inputs["input_ids"][:, -max_tokens:]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, -max_tokens:]

    return inputs


# ============================================================
# Model loading
# Same model/runtime path as adaptive evaluator
# ============================================================

def get_dtype() -> torch.dtype:
    if DTYPE_NAME == "float16":
        return torch.float16
    if DTYPE_NAME == "bfloat16":
        return torch.bfloat16
    if DTYPE_NAME == "float32":
        return torch.float32
    raise ValueError(f"Unsupported DTYPE_NAME: {DTYPE_NAME}")


def get_device() -> torch.device:
    if DEVICE == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")

    print("WARNING: requested MPS, but MPS is not available. Falling back to CPU.")
    return torch.device("cpu")


def load_ministral3(
    device: torch.device,
    dtype: torch.dtype,
    use_attention_entropy: bool,
):
    tokenizer = MistralCommonBackend.from_pretrained(
        MODEL_NAME_OR_PATH,
    )

    kwargs = dict(
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if use_attention_entropy:
        kwargs["attn_implementation"] = "eager"

    try:
        model = Mistral3ForConditionalGeneration.from_pretrained(
            MODEL_NAME_OR_PATH,
            **kwargs,
        )
    except TypeError:
        kwargs.pop("attn_implementation", None)
        model = Mistral3ForConditionalGeneration.from_pretrained(
            MODEL_NAME_OR_PATH,
            **kwargs,
        )

    model.to(device)
    model.eval()

    return tokenizer, model


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    question: str,
    temperature: float,
    device: torch.device,
    max_prompt_tokens: int,
    generation_seed: int | None = None,
) -> str:
    inputs = encode_messages(
        tokenizer=tokenizer,
        messages=make_messages(question),
        device=device,
        max_tokens=max_prompt_tokens,
    )

    input_len = inputs["input_ids"].shape[-1]
    do_sample = float(temperature) > 0.0

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=do_sample,
        use_cache=True,
    )

    if do_sample:
        gen_kwargs["temperature"] = max(float(temperature), 1e-5)
        gen_kwargs["top_p"] = 1.0

    seed_generation(generation_seed)
    output = model.generate(**gen_kwargs)[0]
    new_tokens = output[input_len:]

    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ============================================================
# IO / progress helpers
# ============================================================

def append_jsonl(path: Path, record: dict[str, Any]) -> None:
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


def load_completed() -> tuple[set[tuple[float, int]], dict[float, dict[str, int]]]:
    done: set[tuple[float, int]] = set()

    stats: dict[float, dict[str, int]] = {
        float(t): {
            "correct": 0,
            "incorrect": 0,
            "unparseable": 0,
            "bad_gt": 0,
            "generation_failed": 0,
        }
        for t in TEMPERATURES
    }

    if not RESULTS_FILE.exists():
        return done, stats

    with RESULTS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "temperature" not in rec or "gsm8k_index" not in rec:
                continue

            t = float(rec["temperature"])
            idx = int(rec["gsm8k_index"])

            done.add((t, idx))

            if t not in stats:
                continue

            if rec.get("ground_truth_bad"):
                stats[t]["bad_gt"] += 1
            elif rec.get("generation_failed"):
                stats[t]["generation_failed"] += 1
            elif rec.get("parsed_answer") is None:
                stats[t]["unparseable"] += 1
            elif rec.get("is_correct"):
                stats[t]["correct"] += 1
            else:
                stats[t]["incorrect"] += 1

    return done, stats


def temp_answered(s: dict[str, int]) -> int:
    return (
        s["correct"]
        + s["incorrect"]
        + s["unparseable"]
        + s["generation_failed"]
    )


def temp_evaluated(s: dict[str, int]) -> int:
    return temp_answered(s) + s["bad_gt"]


def temp_accuracy(s: dict[str, int]) -> tuple[float, int, int]:
    answered = temp_answered(s)
    acc = s["correct"] / answered if answered else 0.0
    return acc, s["correct"], answered


def total_runs_done(stats: dict[float, dict[str, int]]) -> int:
    return sum(temp_evaluated(s) for s in stats.values())


# ============================================================
# Main eval
# ============================================================

def main() -> None:
    global TEMPERATURES, MAX_PROMPT_TOKENS, USE_ATTENTION_ENTROPY

    torch.manual_seed(SEED)

    if USE_CKPT_CONFIG_FOR_COMPARABILITY and CONFIG_FILE.exists():
        cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        if CLASSIC_TEMPERATURES_OVERRIDE is None:
            TEMPERATURES = [float(x) for x in cfg.get("temperature_grid", TEMPERATURES)]
        else:
            TEMPERATURES = [float(x) for x in CLASSIC_TEMPERATURES_OVERRIDE]
        MAX_PROMPT_TOKENS = int(cfg.get("max_prompt_tokens", MAX_PROMPT_TOKENS))
        USE_ATTENTION_ENTROPY = bool(cfg.get("use_attention_entropy", USE_ATTENTION_ENTROPY))
    elif CLASSIC_TEMPERATURES_OVERRIDE is not None:
        TEMPERATURES = [float(x) for x in CLASSIC_TEMPERATURES_OVERRIDE]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    dtype = get_dtype()

    print(f"Device       : {device}")
    print(f"Model        : {MODEL_NAME_OR_PATH}")
    print(f"dtype        : {dtype}")
    print(f"Split        : {DATASET_SPLIT}")
    print(f"Temperatures : {TEMPERATURES}")
    print(f"Max prompt tokens: {MAX_PROMPT_TOKENS}")
    print(f"Use eager attention path: {USE_ATTENTION_ENTROPY}")
    print()

    tokenizer, model = load_ministral3(
        device=device,
        dtype=dtype,
        use_attention_entropy=USE_ATTENTION_ENTROPY,
    )

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)

    if NUM_EXAMPLES is not None:
        dataset = dataset.select(range(min(NUM_EXAMPLES, len(dataset))))

    total = len(dataset)
    total_runs = total * len(TEMPERATURES)

    done, stats = load_completed()
    runs_done_at_start = total_runs_done(stats)

    print(f"Examples     : {total}")
    print(f"Total runs   : {total_runs}")
    print(
        f"Resuming     : {runs_done_at_start}/{total_runs} already complete "
        f"({100 * runs_done_at_start / total_runs:.1f}%)"
    )
    print()

    script_start = time.perf_counter()

    for temp in TEMPERATURES:
        t = float(temp)
        temp_start = time.perf_counter()

        print(f"--- temperature = {t} ---")

        for idx in range(total):
            if (t, idx) in done:
                continue

            row = dataset[idx]
            question = row["question"]
            gt = extract_ground_truth(row["answer"])

            if gt is None:
                append_jsonl(
                    RESULTS_FILE,
                    {
                        "temperature": t,
                        "gsm8k_index": idx,
                        "ground_truth_bad": True,
                        "raw_answer": row["answer"],
                    },
                )
                stats[t]["bad_gt"] += 1
                continue

            response = None
            parse_status = "generation_failed"

            for attempt in range(MAX_RETRIES):
                try:
                    response = generate_answer(
                        model=model,
                        tokenizer=tokenizer,
                        question=question,
                        temperature=t,
                        device=device,
                        max_prompt_tokens=MAX_PROMPT_TOKENS,
                        generation_seed=SEED + idx,
                    )
                    break

                except RuntimeError as e:
                    print(
                        f"  [T={t} idx={idx}] runtime error "
                        f"attempt {attempt + 1}/{MAX_RETRIES}: {e}"
                    )

                    if device.type == "mps":
                        torch.mps.empty_cache()

                    time.sleep(2)

                except Exception as e:
                    print(
                        f"  [T={t} idx={idx}] error "
                        f"attempt {attempt + 1}/{MAX_RETRIES}: {e}"
                    )
                    time.sleep(2)

            if response is None:
                append_jsonl(
                    RESULTS_FILE,
                    {
                        "temperature": t,
                        "gsm8k_index": idx,
                        "question": question,
                        "ground_truth": str(gt),
                        "generation_failed": True,
                        "response": None,
                        "parsed_answer": None,
                        "parse_status": parse_status,
                        "is_correct": False,
                    },
                )

                stats[t]["generation_failed"] += 1
                continue

            parsed, parse_status = extract_model_answer(response)

            if parsed is None:
                is_correct = False
                stats[t]["unparseable"] += 1
            else:
                is_correct = numerically_equal(parsed, gt)
                if is_correct:
                    stats[t]["correct"] += 1
                else:
                    stats[t]["incorrect"] += 1

            append_jsonl(
                RESULTS_FILE,
                {
                    "temperature": t,
                    "gsm8k_index": idx,
                    "question": question,
                    "ground_truth": str(gt),
                    "response": response,
                    "parsed_answer": str(parsed) if parsed is not None else None,
                    "parse_status": parse_status,
                    "is_correct": is_correct,
                },
            )

            acc, correct, answered = temp_accuracy(stats[t])
            mark = "✓" if is_correct else ("?" if parsed is None else "✗")

            elapsed = time.perf_counter() - script_start
            runs_done_now = total_runs_done(stats)
            session_runs = runs_done_now - runs_done_at_start
            rate = session_runs / elapsed if elapsed > 0 and session_runs > 0 else 0.0
            remaining = total_runs - runs_done_now
            eta = remaining / rate if rate > 0 else 0.0
            overall_pct = 100 * runs_done_now / total_runs

            print(
                f"  [T={t} {idx + 1}/{total}] {mark} "
                f"acc={acc:.4f} ({correct}/{answered}) | "
                f"overall {runs_done_now}/{total_runs} ({overall_pct:.1f}%) | "
                f"elapsed={fmt_time(elapsed)} | "
                f"eta={fmt_time(eta)}"
            )

        temp_elapsed = time.perf_counter() - temp_start
        acc, correct, answered = temp_accuracy(stats[t])

        print(
            f"  >> T={t} done in {fmt_time(temp_elapsed)}: "
            f"accuracy={acc:.4f} ({correct}/{answered})"
        )
        print()

    total_elapsed = time.perf_counter() - script_start

    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append(MODEL_NAME_OR_PATH)
    summary_lines.append("Classic fixed-temperature HF evaluation")
    summary_lines.append("")

    for temp in TEMPERATURES:
        t = float(temp)
        acc, correct, answered = temp_accuracy(stats[t])
        summary_lines.append(
            f"  T={t:<4} -> accuracy = {acc:.4f}  ({correct}/{answered})"
        )

    summary_lines.append("")
    summary_lines.append(f"Total runtime: {fmt_time(total_elapsed)}")

    summary_text = "\n".join(summary_lines)

    print(summary_text)

    SUMMARY_TXT_FILE.write_text(summary_text + "\n", encoding="utf-8")

    summary_json = {
        "model": MODEL_NAME_OR_PATH,
        "dataset": f"{DATASET_NAME}/{DATASET_CONFIG}",
        "split": DATASET_SPLIT,
        "num_examples": total,
        "seed": SEED,
        "temperatures": [float(t) for t in TEMPERATURES],
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "dtype": DTYPE_NAME,
        "device": str(device),
        "use_attention_entropy": USE_ATTENTION_ENTROPY,
        "attention_implementation_requested": "eager" if USE_ATTENTION_ENTROPY else "default",
        "generation_seed_policy": "torch.manual_seed(SEED + gsm8k_index) immediately before model.generate",
        "used_adaptive_config_for_comparability": USE_CKPT_CONFIG_FOR_COMPARABILITY and CONFIG_FILE.exists(),
        "total_runtime_seconds": total_elapsed,
        "results": [
            {
                "temperature": float(t),
                "accuracy": temp_accuracy(stats[float(t)])[0],
                "correct": stats[float(t)]["correct"],
                "incorrect": stats[float(t)]["incorrect"],
                "unparseable": stats[float(t)]["unparseable"],
                "generation_failed": stats[float(t)]["generation_failed"],
                "bad_ground_truth": stats[float(t)]["bad_gt"],
                "answered": temp_accuracy(stats[float(t)])[2],
            }
            for t in TEMPERATURES
        ],
        "note": (
            "Classic fixed-temperature baseline using the same Hugging Face "
            "model, tokenizer, chat template, prompt, generation path, and "
            "answer parser as evaluate_adaptive_decoder.py. The only intended "
            "experimental difference is fixed temperature vs adaptive "
            "per-sequence temperature selection."
        ),
    }

    SUMMARY_JSON_FILE.write_text(
        json.dumps(summary_json, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Summary saved to: {SUMMARY_TXT_FILE}")
    print(f"JSON saved to   : {SUMMARY_JSON_FILE}")
    print(f"Raw results     : {RESULTS_FILE}")


if __name__ == "__main__":
    main()