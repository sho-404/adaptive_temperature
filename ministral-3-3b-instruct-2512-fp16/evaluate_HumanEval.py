from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend


# ============================================================
# CONFIG - edit here, not through CLI args
# ============================================================

MODEL_NAME_OR_PATH = "mistralai/Ministral-3-3B-Instruct-2512-BF16"

# Read adaptive checkpoint config so classic uses same prompt/token/grid settings.
CKPT_DIR = Path("adaptive_decoder_ckpt")
CONFIG_FILE = CKPT_DIR / "config.json"

DEVICE = "mps"
DTYPE_NAME = "float16"

# Hugging Face mirror of OpenAI HumanEval
DATASET_NAME = "openai/openai_humaneval"
DATASET_SPLIT = "test"
NUM_EXAMPLES = 4  # None = full HumanEval, usually 164 tasks

OUTPUT_DIR = Path("humaneval_full_n100_classic_hf")
RESULTS_FILE = OUTPUT_DIR / "results.jsonl"
SUMMARY_TXT_FILE = OUTPUT_DIR / "summary.txt"
SUMMARY_JSON_FILE = OUTPUT_DIR / "summary.json"

# HumanEval sampling settings
N_SAMPLES_PER_TASK = 10
PASS_AT_K_VALUES = [1, 10, 100]

# Same idea as your GSM8K classic baseline
TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

USE_CKPT_CONFIG_FOR_COMPARABILITY = True
CLASSIC_TEMPERATURES_OVERRIDE = None  # example: [0.2, 0.4, 0.6, 0.8, 1.0]

MAX_PROMPT_TOKENS = 768
MAX_NEW_TOKENS = 512
SEED = 42
MAX_RETRIES = 2

# HumanEval execution
EXECUTION_TIMEOUT_SECONDS = 5

# Strict comparability with your adaptive path
USE_ATTENTION_ENTROPY = True


# ============================================================
# Seeding
# ============================================================

def seed_generation(seed: int | None) -> None:
    """Seed immediately before generate() so results do not depend on loop order."""
    if seed is None:
        return

    torch.manual_seed(int(seed))

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# ============================================================
# Prompt / tokenizer helpers
# ============================================================

def make_messages(prompt: str) -> list[dict[str, str]]:
    system = (
        "You are a careful Python coding assistant. "
        "You write correct, minimal Python code. "
        "Return only code. Do not use markdown fences."
    )

    user = f"""Complete the following Python function.

Rules:
1. Return only the code needed to complete the function.
2. Do not include markdown fences.
3. Do not explain the solution.
4. Do not redefine the function unless necessary.
5. Preserve the given function signature.

Prompt:
{prompt}"""

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
# Same model/runtime path as your GSM8K evaluator
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

    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    print(f"WARNING: requested {DEVICE}, but it is not available. Falling back to CPU.")
    return torch.device("cpu")


from huggingface_hub import snapshot_download

def load_ministral3(device, dtype, use_attention_entropy):
    local_path = snapshot_download(MODEL_NAME_OR_PATH, local_files_only=True)

    tokenizer = MistralCommonBackend.from_pretrained(local_path)

    kwargs = dict(dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)
    if use_attention_entropy:
        kwargs["attn_implementation"] = "eager"

    try:
        model = Mistral3ForConditionalGeneration.from_pretrained(local_path, **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        model = Mistral3ForConditionalGeneration.from_pretrained(local_path, **kwargs)

    model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate_completion(
    model,
    tokenizer,
    prompt: str,
    temperature: float,
    device: torch.device,
    max_prompt_tokens: int,
    generation_seed: int | None = None,
) -> str:
    inputs = encode_messages(
        tokenizer=tokenizer,
        messages=make_messages(prompt),
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
# HumanEval cleaning / execution
# ============================================================

def strip_markdown_fences(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:python)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    return text.strip()


def truncate_after_extra_content(code: str) -> str:
    """
    Keep code only. This removes common chatty endings while preserving Python.
    Conservative by design.
    """
    code = strip_markdown_fences(code)

    stop_markers = [
        "\n# Explanation",
        "\nExplanation:",
        "\nThe solution",
        "\nThis function",
        "\n```",
    ]

    for marker in stop_markers:
        if marker in code:
            code = code.split(marker)[0].rstrip()

    return code.rstrip() + "\n"


def indent_completion_if_needed(prompt: str, completion: str) -> str:
    """
    HumanEval prompt usually ends inside a function docstring/signature block.
    Chat models may return either indented function body or a full function.
    """
    completion = truncate_after_extra_content(completion)

    # If model redefines function, keep as-is.
    if re.search(r"(?m)^def\s+\w+\s*\(", completion):
        return completion

    # If completion already appears indented, keep as-is.
    nonempty = [line for line in completion.splitlines() if line.strip()]
    if nonempty and all(line.startswith((" ", "\t")) for line in nonempty[: min(3, len(nonempty))]):
        return completion

    # Otherwise indent it as function body.
    return textwrap.indent(completion, "    ")


def build_humaneval_program(row: dict[str, Any], completion: str) -> str:
    prompt = row["prompt"]
    test = row["test"]
    entry_point = row["entry_point"]

    fixed_completion = indent_completion_if_needed(prompt, completion)

    program = (
        prompt
        + fixed_completion
        + "\n"
        + test
        + "\n"
        + f"check({entry_point})\n"
    )

    return program


def run_program_in_subprocess(program: str, timeout_seconds: int) -> tuple[bool, str]:
    """
    Executes generated code in a separate Python process.

    Warning:
    This is still not a true security sandbox. Use a disposable environment.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        program_path = Path(tmpdir) / "candidate.py"
        program_path.write_text(program, encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONPATH"] = tmpdir

        try:
            proc = subprocess.run(
                ["python", str(program_path)],
                cwd=tmpdir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                text=True,
            )

            passed = proc.returncode == 0
            output = ""

            if proc.stdout:
                output += proc.stdout[-4000:]

            if proc.stderr:
                output += proc.stderr[-4000:]

            return passed, output.strip()

        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout_seconds}s"

        except Exception as e:
            return False, repr(e)


# ============================================================
# pass@k
# ============================================================

def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    OpenAI HumanEval unbiased pass@k estimator.

    n = number of generated samples
    c = number of correct samples
    k = pass@k value

    If n == k, this becomes 1.0 if c > 0 else 0.0.
    """
    if n < k:
        return float("nan")

    if c == 0:
        return 0.0

    if n - c < k:
        return 1.0

    return 1.0 - math.prod(1.0 - k / i for i in range(n - c + 1, n + 1))


def aggregate_pass_at_k(records: list[dict[str, Any]], temperatures: list[float]) -> dict[float, dict[str, Any]]:
    by_temp_task: dict[float, dict[str, list[dict[str, Any]]]] = {
        float(t): {} for t in temperatures
    }

    for rec in records:
        if rec.get("generation_failed"):
            continue

        t = float(rec["temperature"])
        task_id = str(rec["task_id"])

        if t not in by_temp_task:
            by_temp_task[t] = {}

        by_temp_task[t].setdefault(task_id, []).append(rec)

    out: dict[float, dict[str, Any]] = {}

    for t, task_map in by_temp_task.items():
        task_scores = []

        for task_id, rows in task_map.items():
            n = len(rows)
            c = sum(1 for r in rows if r.get("passed") is True)

            task_result = {
                "task_id": task_id,
                "n": n,
                "correct": c,
            }

            for k in PASS_AT_K_VALUES:
                task_result[f"pass@{k}"] = estimate_pass_at_k(n=n, c=c, k=int(k))

            task_scores.append(task_result)

        summary = {
            "num_tasks": len(task_scores),
            "samples_per_task_target": N_SAMPLES_PER_TASK,
            "pass_at_k": {},
            "total_correct_samples": sum(x["correct"] for x in task_scores),
            "total_samples": sum(x["n"] for x in task_scores),
        }

        for k in PASS_AT_K_VALUES:
            vals = [
                x[f"pass@{k}"]
                for x in task_scores
                if not math.isnan(float(x[f"pass@{k}"]))
            ]
            summary["pass_at_k"][f"pass@{k}"] = sum(vals) / len(vals) if vals else float("nan")

        out[t] = {
            "summary": summary,
            "tasks": task_scores,
        }

    return out


# ============================================================
# IO / progress helpers
# ============================================================

def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return records


def fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)

    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def load_completed() -> set[tuple[float, str, int]]:
    done: set[tuple[float, str, int]] = set()

    for rec in read_jsonl(RESULTS_FILE):
        if "temperature" not in rec or "task_id" not in rec or "sample_index" not in rec:
            continue

        done.add(
            (
                float(rec["temperature"]),
                str(rec["task_id"]),
                int(rec["sample_index"]),
            )
        )

    return done


def total_runs_done(done: set[tuple[float, str, int]]) -> int:
    return len(done)


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
    print(f"Dataset      : {DATASET_NAME}")
    print(f"Split        : {DATASET_SPLIT}")
    print(f"Temperatures : {TEMPERATURES}")
    print(f"Samples/task : {N_SAMPLES_PER_TASK}")
    print(f"pass@k       : {PASS_AT_K_VALUES}")
    print(f"Max prompt tokens: {MAX_PROMPT_TOKENS}")
    print(f"Max new tokens   : {MAX_NEW_TOKENS}")
    print(f"Use eager attention path: {USE_ATTENTION_ENTROPY}")
    print()

    tokenizer, model = load_ministral3(
        device=device,
        dtype=dtype,
        use_attention_entropy=USE_ATTENTION_ENTROPY,
    )

    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    if NUM_EXAMPLES is not None:
        dataset = dataset.select(range(min(NUM_EXAMPLES, len(dataset))))

    total_tasks = len(dataset)
    total_runs = total_tasks * len(TEMPERATURES) * N_SAMPLES_PER_TASK

    done = load_completed()
    runs_done_at_start = total_runs_done(done)

    print(f"Tasks        : {total_tasks}")
    print(f"Total runs   : {total_runs}")
    print(
        f"Resuming     : {runs_done_at_start}/{total_runs} already complete "
        f"({100 * runs_done_at_start / total_runs:.1f}%)"
    )
    print()

    if 0.0 in [float(t) for t in TEMPERATURES] and N_SAMPLES_PER_TASK > 1:
        print(
            "NOTE: T=0.0 uses greedy decoding, so its 100 samples will usually be duplicates. "
            "That is okay for baseline logging, but pass@100 is only meaningful with sampling temperatures."
        )
        print()

    script_start = time.perf_counter()

    for temp in TEMPERATURES:
        t = float(temp)
        temp_start = time.perf_counter()

        print(f"--- temperature = {t} ---")

        for task_idx in range(total_tasks):
            row = dataset[task_idx]
            task_id = str(row["task_id"])
            prompt = row["prompt"]

            for sample_idx in range(N_SAMPLES_PER_TASK):
                key = (t, task_id, sample_idx)

                if key in done:
                    continue

                response = None
                generation_failed = False
                error_text = None

                for attempt in range(MAX_RETRIES):
                    try:
                        response = generate_completion(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            temperature=t,
                            device=device,
                            max_prompt_tokens=MAX_PROMPT_TOKENS,
                            generation_seed=SEED + task_idx * N_SAMPLES_PER_TASK + sample_idx,
                        )
                        break

                    except RuntimeError as e:
                        print(
                            f"  [T={t} task={task_id} sample={sample_idx}] runtime error "
                            f"attempt {attempt + 1}/{MAX_RETRIES}: {e}"
                        )

                        if device.type == "mps":
                            torch.mps.empty_cache()

                        time.sleep(2)

                    except Exception as e:
                        print(
                            f"  [T={t} task={task_id} sample={sample_idx}] error "
                            f"attempt {attempt + 1}/{MAX_RETRIES}: {e}"
                        )
                        time.sleep(2)

                if response is None:
                    generation_failed = True
                    error_text = "generation_failed"

                    append_jsonl(
                        RESULTS_FILE,
                        {
                            "temperature": t,
                            "task_index": task_idx,
                            "task_id": task_id,
                            "sample_index": sample_idx,
                            "prompt": prompt,
                            "completion": None,
                            "program": None,
                            "passed": False,
                            "generation_failed": True,
                            "error": error_text,
                        },
                    )

                    done.add(key)
                    continue

                cleaned_completion = indent_completion_if_needed(prompt, response)
                program = build_humaneval_program(row, response)
                passed, execution_output = run_program_in_subprocess(
                    program=program,
                    timeout_seconds=EXECUTION_TIMEOUT_SECONDS,
                )

                append_jsonl(
                    RESULTS_FILE,
                    {
                        "temperature": t,
                        "task_index": task_idx,
                        "task_id": task_id,
                        "sample_index": sample_idx,
                        "prompt": prompt,
                        "raw_response": response,
                        "completion": cleaned_completion,
                        "passed": passed,
                        "generation_failed": generation_failed,
                        "execution_output": execution_output,
                    },
                )

                done.add(key)

                elapsed = time.perf_counter() - script_start
                runs_done_now = total_runs_done(done)
                session_runs = runs_done_now - runs_done_at_start
                rate = session_runs / elapsed if elapsed > 0 and session_runs > 0 else 0.0
                remaining = total_runs - runs_done_now
                eta = remaining / rate if rate > 0 else 0.0
                overall_pct = 100 * runs_done_now / total_runs

                mark = "✓" if passed else "✗"

                print(
                    f"  [T={t} task={task_idx + 1}/{total_tasks} {task_id} "
                    f"sample={sample_idx + 1}/{N_SAMPLES_PER_TASK}] {mark} | "
                    f"overall {runs_done_now}/{total_runs} ({overall_pct:.1f}%) | "
                    f"elapsed={fmt_time(elapsed)} | eta={fmt_time(eta)}"
                )

        temp_elapsed = time.perf_counter() - temp_start

        records_now = read_jsonl(RESULTS_FILE)
        agg_now = aggregate_pass_at_k(records_now, TEMPERATURES)

        if t in agg_now:
            pass_summary = agg_now[t]["summary"]["pass_at_k"]
            pass_text = " | ".join(
                f"{k}={v:.4f}" for k, v in pass_summary.items()
            )
            print(f"  >> T={t} done in {fmt_time(temp_elapsed)}: {pass_text}")
        else:
            print(f"  >> T={t} done in {fmt_time(temp_elapsed)}")

        print()

    total_elapsed = time.perf_counter() - script_start

    all_records = read_jsonl(RESULTS_FILE)
    aggregate = aggregate_pass_at_k(all_records, TEMPERATURES)

    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append(MODEL_NAME_OR_PATH)
    summary_lines.append("Classic fixed-temperature HumanEval evaluation")
    summary_lines.append("")
    summary_lines.append(f"Dataset: {DATASET_NAME}/{DATASET_SPLIT}")
    summary_lines.append(f"Tasks: {total_tasks}")
    summary_lines.append(f"Samples per task: {N_SAMPLES_PER_TASK}")
    summary_lines.append("")

    for temp in TEMPERATURES:
        t = float(temp)

        if t not in aggregate:
            continue

        pass_summary = aggregate[t]["summary"]["pass_at_k"]

        summary_lines.append(f"Temperature {t}:")
        for k, v in pass_summary.items():
            summary_lines.append(f"  {k:<8} = {v:.4f}")

        total_correct = aggregate[t]["summary"]["total_correct_samples"]
        total_samples = aggregate[t]["summary"]["total_samples"]
        sample_acc = total_correct / total_samples if total_samples else 0.0

        summary_lines.append(f"  sample_acc = {sample_acc:.4f} ({total_correct}/{total_samples})")
        summary_lines.append("")

    summary_lines.append(f"Total runtime: {fmt_time(total_elapsed)}")

    summary_text = "\n".join(summary_lines)

    print(summary_text)

    SUMMARY_TXT_FILE.write_text(summary_text + "\n", encoding="utf-8")

    summary_json = {
        "model": MODEL_NAME_OR_PATH,
        "dataset": DATASET_NAME,
        "split": DATASET_SPLIT,
        "num_tasks": total_tasks,
        "samples_per_task": N_SAMPLES_PER_TASK,
        "pass_at_k_values": PASS_AT_K_VALUES,
        "seed": SEED,
        "temperatures": [float(t) for t in TEMPERATURES],
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "dtype": DTYPE_NAME,
        "device": str(device),
        "use_attention_entropy": USE_ATTENTION_ENTROPY,
        "attention_implementation_requested": "eager" if USE_ATTENTION_ENTROPY else "default",
        "generation_seed_policy": (
            "torch.manual_seed(SEED + task_idx * N_SAMPLES_PER_TASK + sample_idx) "
            "immediately before model.generate"
        ),
        "used_adaptive_config_for_comparability": (
            USE_CKPT_CONFIG_FOR_COMPARABILITY and CONFIG_FILE.exists()
        ),
        "execution_timeout_seconds": EXECUTION_TIMEOUT_SECONDS,
        "total_runtime_seconds": total_elapsed,
        "results": aggregate,
        "note": (
            "Classic fixed-temperature HumanEval baseline using the same Hugging Face "
            "model, tokenizer, chat template, model loading path, dtype/device config, "
            "and temperature-grid comparability logic as the GSM8K evaluator. "
            "The task-specific logic is changed from numeric answer parsing to Python "
            "code generation plus functional correctness execution."
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