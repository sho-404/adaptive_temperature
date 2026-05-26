from __future__ import annotations

import json
import os
import re
import time
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

getcontext().prec = 50

# ============================================================
# CONFIG - edit here, not through CLI args
# ============================================================

MODEL_NAME_OR_PATH = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
CKPT_DIR = Path("adaptive_decoder_ckpt")
CKPT_FILE = CKPT_DIR / "adaptive_temperature_mlp.pt"
CONFIG_FILE = CKPT_DIR / "config.json"

DEVICE = "mps"
DTYPE_NAME = "float16"

DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "test"
NUM_EXAMPLES = None

OUTPUT_DIR = Path("gsm8k_eval_adaptive_hf")
RESULTS_FILE = OUTPUT_DIR / "results.jsonl"
SUMMARY_TXT_FILE = OUTPUT_DIR / "summary.txt"
SUMMARY_JSON_FILE = OUTPUT_DIR / "summary.json"

MAX_PROMPT_TOKENS = 768
MAX_NEW_TOKENS = 512
SEED = 42
MAX_RETRIES = 2
ABS_TOL = Decimal("1e-6")
ALLOW_LAST_NUMBER_FALLBACK = False

TEMPERATURE_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
MLP_HIDDEN_DIM = 2048
DROPOUT = 0.10
USE_ATTENTION_ENTROPY = True

# ============================================================
# Parsing
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
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*-?\d[\d,]*(?:\.\d+)?)?", response)
        if nums:
            value = parse_decimal(nums[-1])
            if value is not None:
                return value, "last_number_fallback"

    return None, "unparseable"


def numerically_equal(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) <= ABS_TOL


def seed_generation(seed: int | None) -> None:
    """Seed immediately before generate() so results do not depend on prior signal extraction or loop order."""
    if seed is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

# ============================================================
# Prompt / tokenizer helpers
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


def encode_messages(tokenizer, messages: list[dict[str, str]], device: torch.device, max_tokens: int) -> dict[str, torch.Tensor]:
    tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)
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
# Model and MLP
# ============================================================

class TemperaturePreferenceMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


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


def load_ministral3(device: torch.device, dtype: torch.dtype, use_attention_entropy: bool):
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
        model = Mistral3ForConditionalGeneration.from_pretrained(MODEL_NAME_OR_PATH, **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        model = Mistral3ForConditionalGeneration.from_pretrained(MODEL_NAME_OR_PATH, **kwargs)

    model.to(device)
    model.eval()
    return tokenizer, model


def safe_forward(model, inputs: dict[str, torch.Tensor], want_attentions: bool):
    try:
        return model(
            **inputs,
            output_hidden_states=True,
            output_attentions=want_attentions,
            use_cache=False,
            return_dict=True,
        )
    except Exception as e:
        if want_attentions:
            print(f"WARNING: attention output failed; retrying without attentions. Original: {type(e).__name__}: {e}")
            return model(
                **inputs,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=False,
                return_dict=True,
            )
        raise


@torch.no_grad()
def extract_prompt_signals(model, tokenizer, question: str, device: torch.device, max_prompt_tokens: int, use_attention_entropy: bool) -> torch.Tensor:
    inputs = encode_messages(tokenizer, make_messages(question), device, max_prompt_tokens)
    out = safe_forward(model, inputs, use_attention_entropy)

    hidden_last = out.hidden_states[-1][:, -1, :].float().squeeze(0).cpu()

    logits = out.logits[:, -1, :].float()
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).item()

    top2_probs = torch.topk(probs, k=2, dim=-1).values.squeeze(0)
    prob_gap = (top2_probs[0] - top2_probs[1]).item()

    top2_logits = torch.topk(logits, k=2, dim=-1).values.squeeze(0)
    logit_gap = (top2_logits[0] - top2_logits[1]).item()

    attentions = getattr(out, "attentions", None)
    if use_attention_entropy and attentions is not None:
        attn = attentions[-1][0, :, -1, :].float().clamp_min(1e-12)
        attn_entropy = (-(attn * attn.log()).sum(dim=-1)).mean().item()
    else:
        attn_entropy = 0.0

    attention_mask = inputs.get("attention_mask")
    seq_len = int(attention_mask.sum().item()) if attention_mask is not None else int(inputs["input_ids"].shape[-1])
    norm_pos = min(seq_len, max_prompt_tokens) / float(max_prompt_tokens)

    scalars = torch.tensor([entropy, prob_gap, logit_gap, attn_entropy, norm_pos], dtype=torch.float32)
    return torch.cat([hidden_last, scalars], dim=0)


def feature_with_temperature(base_features: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.cat([base_features, torch.tensor([float(tau)], dtype=torch.float32)], dim=0)


@torch.no_grad()
def choose_temperature(mlp, base_features: torch.Tensor, grid: list[float], device: torch.device) -> tuple[float, dict[str, float]]:
    xs = torch.stack([feature_with_temperature(base_features, t) for t in grid], dim=0).to(device)
    scores = mlp(xs).float().cpu()
    best_i = int(scores.argmax().item())
    score_map = {str(float(t)): float(scores[i].item()) for i, t in enumerate(grid)}
    return float(grid[best_i]), score_map


@torch.no_grad()
def generate_answer(model, tokenizer, question: str, temperature: float, device: torch.device, max_prompt_tokens: int, generation_seed: int | None = None) -> str:
    inputs = encode_messages(tokenizer, make_messages(question), device, max_prompt_tokens)
    input_len = inputs["input_ids"].shape[-1]
    do_sample = temperature > 0.0

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
# IO / progress
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


def load_completed() -> tuple[set[int], dict[str, int]]:
    done: set[int] = set()
    stats = {"correct": 0, "incorrect": 0, "unparseable": 0, "bad_gt": 0, "generation_failed": 0}
    if not RESULTS_FILE.exists():
        return done, stats
    with RESULTS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = int(rec["gsm8k_index"])
            done.add(idx)
            if rec.get("ground_truth_bad"):
                stats["bad_gt"] += 1
            elif rec.get("generation_failed"):
                stats["generation_failed"] += 1
            elif rec.get("parsed_answer") is None:
                stats["unparseable"] += 1
            elif rec.get("is_correct"):
                stats["correct"] += 1
            else:
                stats["incorrect"] += 1
    return done, stats


def evaluated_count(stats: dict[str, int]) -> int:
    return stats["correct"] + stats["incorrect"] + stats["unparseable"] + stats["generation_failed"] + stats["bad_gt"]


def answered(stats: dict[str, int]) -> int:
    return stats["correct"] + stats["incorrect"] + stats["unparseable"] + stats["generation_failed"]


def accuracy(stats: dict[str, int]) -> float:
    denom = answered(stats)
    return stats["correct"] / denom if denom else 0.0

# ============================================================
# Main eval
# ============================================================

def main() -> None:
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    dtype = get_dtype()

    if not CKPT_FILE.exists():
        raise RuntimeError(f"Missing checkpoint: {CKPT_FILE}. Run train_adaptive_decoder_mistral.py first.")

    cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8")) if CONFIG_FILE.exists() else {}
    grid = [float(x) for x in cfg.get("temperature_grid", TEMPERATURE_GRID)]
    input_dim = int(cfg.get("input_dim", 0))
    hidden_dim = int(cfg.get("mlp_hidden_dim", MLP_HIDDEN_DIM))
    dropout = float(cfg.get("dropout", DROPOUT))
    max_prompt_tokens = int(cfg.get("max_prompt_tokens", MAX_PROMPT_TOKENS))
    use_attention_entropy = bool(cfg.get("use_attention_entropy", USE_ATTENTION_ENTROPY))

    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME_OR_PATH}")
    print(f"Checkpoint: {CKPT_FILE}")
    print(f"Temperature grid: {grid}")
    print(f"Max prompt tokens: {max_prompt_tokens}")
    print(f"Use attention entropy / eager attention path: {use_attention_entropy}")

    tokenizer, model = load_ministral3(device, dtype, use_attention_entropy)

    if input_dim <= 0:
        cfg_model = getattr(model.config, "text_config", None) or model.config
        hidden_size = int(getattr(cfg_model, "hidden_size"))
        input_dim = hidden_size + 5 + 1

    mlp = TemperaturePreferenceMLP(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    ckpt = torch.load(CKPT_FILE, map_location=device)
    mlp.load_state_dict(ckpt["mlp_state_dict"])
    mlp.eval()

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    if NUM_EXAMPLES is not None:
        dataset = dataset.select(range(min(NUM_EXAMPLES, len(dataset))))
    total = len(dataset)

    done, stats = load_completed()
    start_evaluated = evaluated_count(stats)
    print(f"Examples: {total}")
    print(f"Resuming: {len(done)}/{total} already complete")
    print()

    start = time.perf_counter()

    for idx in range(total):
        if idx in done:
            continue

        row = dataset[idx]
        question = row["question"]
        gt = extract_ground_truth(row["answer"])

        if gt is None:
            append_jsonl(RESULTS_FILE, {"gsm8k_index": idx, "ground_truth_bad": True, "raw_answer": row["answer"]})
            stats["bad_gt"] += 1
            continue

        response = None
        selected_temp = None
        score_map = None
        parse_status = "generation_failed"

        for attempt in range(MAX_RETRIES):
            try:
                base = extract_prompt_signals(model, tokenizer, question, device, max_prompt_tokens, use_attention_entropy)
                selected_temp, score_map = choose_temperature(mlp, base, grid, device)
                response = generate_answer(model, tokenizer, question, selected_temp, device, max_prompt_tokens, generation_seed=SEED + idx)
                break
            except RuntimeError as e:
                print(f"  [idx={idx}] runtime error attempt {attempt+1}/{MAX_RETRIES}: {e}")
                if device.type == "mps":
                    torch.mps.empty_cache()
                time.sleep(2)
            except Exception as e:
                print(f"  [idx={idx}] error attempt {attempt+1}/{MAX_RETRIES}: {e}")
                time.sleep(2)

        if response is None:
            append_jsonl(RESULTS_FILE, {
                "gsm8k_index": idx,
                "question": question,
                "ground_truth": str(gt),
                "generation_failed": True,
                "selected_temperature": selected_temp,
                "temperature_scores": score_map,
                "response": None,
                "parsed_answer": None,
                "parse_status": parse_status,
                "is_correct": False,
            })
            stats["generation_failed"] += 1
            continue

        parsed, parse_status = extract_model_answer(response)
        if parsed is None:
            is_correct = False
            stats["unparseable"] += 1
        else:
            is_correct = numerically_equal(parsed, gt)
            if is_correct:
                stats["correct"] += 1
            else:
                stats["incorrect"] += 1

        append_jsonl(RESULTS_FILE, {
            "gsm8k_index": idx,
            "question": question,
            "ground_truth": str(gt),
            "selected_temperature": selected_temp,
            "temperature_scores": score_map,
            "response": response,
            "parsed_answer": str(parsed) if parsed is not None else None,
            "parse_status": parse_status,
            "is_correct": is_correct,
        })

        elapsed = time.perf_counter() - start
        session_done = evaluated_count(stats) - start_evaluated
        rate = session_done / elapsed if elapsed > 0 and session_done > 0 else 0.0
        remaining = total - (len(done) + session_done)
        eta = remaining / rate if rate > 0 else 0.0
        mark = "✓" if is_correct else ("?" if parsed is None else "✗")
        print(
            f"[{idx+1}/{total}] {mark} temp={selected_temp} "
            f"acc={accuracy(stats):.4f} ({stats['correct']}/{answered(stats)}) "
            f"elapsed={fmt_time(elapsed)} eta={fmt_time(eta)}"
        )

    elapsed = time.perf_counter() - start
    summary_text = "\n".join([
        "=" * 60,
        MODEL_NAME_OR_PATH,
        "Adaptive per-sequence temperature evaluation",
        f"Accuracy: {accuracy(stats):.4f} ({stats['correct']}/{answered(stats)})",
        f"Incorrect: {stats['incorrect']}",
        f"Unparseable: {stats['unparseable']}",
        f"Generation failed: {stats['generation_failed']}",
        f"Bad GT: {stats['bad_gt']}",
        f"Total runtime: {fmt_time(elapsed)}",
    ])
    print("\n" + summary_text)
    SUMMARY_TXT_FILE.write_text(summary_text + "\n", encoding="utf-8")
    SUMMARY_JSON_FILE.write_text(json.dumps({
        "model": MODEL_NAME_OR_PATH,
        "dataset": f"{DATASET_NAME}/{DATASET_CONFIG}",
        "split": DATASET_SPLIT,
        "num_examples": total,
        "accuracy": accuracy(stats),
        "stats": stats,
        "temperature_grid": grid,
        "max_prompt_tokens": max_prompt_tokens,
        "seed": SEED,
        "use_attention_entropy": use_attention_entropy,
        "attention_implementation_requested": "eager" if use_attention_entropy else "default",
        "generation_seed_policy": "torch.manual_seed(SEED + gsm8k_index) immediately before model.generate",
        "runtime_seconds": elapsed,
        "note": "Per-sequence adaptive temperature. One chosen temperature per GSM8K question. Generation path is matched to classic_comparable except for temperature selection.",
    }, indent=2), encoding="utf-8")

    print(f"Summary saved to: {SUMMARY_TXT_FILE}")
    print(f"JSON saved to   : {SUMMARY_JSON_FILE}")
    print(f"Raw results     : {RESULTS_FILE}")


if __name__ == "__main__":
    main()
