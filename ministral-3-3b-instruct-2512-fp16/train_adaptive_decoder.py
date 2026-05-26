from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

# ============================================================
# CONFIG - edit here, not through CLI args
# ============================================================

MODEL_NAME_OR_PATH = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
DATASET_PATH = "gsm8k_lpo_ollama/preference_pairs.jsonl"
OUTPUT_DIR = Path("adaptive_decoder_ckpt")

DEVICE = "mps"
DTYPE_NAME = "float16"

SEED = 42
EPOCHS = 2
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
MLP_HIDDEN_DIM = 2048
DROPOUT = 0.10
MARGIN = 0.0
MAX_PROMPT_TOKENS = 768

TEMPERATURE_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Closest to the paper = True. If MPS memory dies, set False.
USE_ATTENTION_ENTROPY = True

SAVE_EVERY_STEPS = 100
LOG_EVERY_STEPS = 10

# ============================================================
# Helpers
# ============================================================

@dataclass
class SavedConfig:
    model_name_or_path: str
    input_dim: int
    hidden_size: int
    mlp_hidden_dim: int
    dropout: float
    max_prompt_tokens: int
    temperature_grid: list[float]
    use_attention_entropy: bool
    dtype_name: str
    note: str


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    if "input_ids" not in inputs:
        raise RuntimeError(f"Tokenizer did not return input_ids. Keys: {list(tokenized.keys())}")

    # Left-truncate if needed. GSM8K prompts are short, but keep this robust.
    seq_len = inputs["input_ids"].shape[-1]
    if seq_len > max_tokens:
        inputs["input_ids"] = inputs["input_ids"][:, -max_tokens:]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, -max_tokens:]

    return inputs


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"

# ============================================================
# MLP
# ============================================================

class TemperaturePreferenceMLP(nn.Module):
    """Per-sequence temperature scorer. It chooses one temperature before generation."""

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

# ============================================================
# Model loading and signal extraction
# ============================================================


def load_ministral3(device: torch.device, dtype: torch.dtype):
    tokenizer = MistralCommonBackend.from_pretrained(
        MODEL_NAME_OR_PATH,
    )
    kwargs = dict(
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if USE_ATTENTION_ENTROPY:
        kwargs["attn_implementation"] = "eager"

    try:
        model = Mistral3ForConditionalGeneration.from_pretrained(MODEL_NAME_OR_PATH, **kwargs)
    except TypeError:
        # Some dev versions may not accept attn_implementation for this class.
        kwargs.pop("attn_implementation", None)
        model = Mistral3ForConditionalGeneration.from_pretrained(MODEL_NAME_OR_PATH, **kwargs)

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
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


def get_hidden_size_from_config(model) -> int:
    cfg = getattr(model.config, "text_config", None) or model.config
    if hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    if hasattr(cfg, "d_model"):
        return int(cfg.d_model)
    raise RuntimeError("Could not infer hidden size from model config.")


@torch.no_grad()
def extract_prompt_signals(model, tokenizer, question: str, device: torch.device) -> torch.Tensor:
    """
    Paper-style signals compressed to prompt-level/per-sequence form:
    1. final-layer hidden state at prompt boundary
    2. next-token entropy
    3. next-token probability gap
    4. next-token logit gap
    5. last-layer attention entropy, if available
    6. normalized prompt length
    """
    inputs = encode_messages(tokenizer, make_messages(question), device, MAX_PROMPT_TOKENS)
    out = safe_forward(model, inputs, USE_ATTENTION_ENTROPY)

    if not hasattr(out, "logits") or out.logits is None:
        raise RuntimeError(f"Model output has no logits. Output type: {type(out)}")
    if not hasattr(out, "hidden_states") or out.hidden_states is None:
        raise RuntimeError(f"Model output has no hidden_states. Output keys: {out.keys() if hasattr(out, 'keys') else None}")

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
    if USE_ATTENTION_ENTROPY and attentions is not None:
        attn = attentions[-1][0, :, -1, :].float().clamp_min(1e-12)
        attn_entropy = (-(attn * attn.log()).sum(dim=-1)).mean().item()
    else:
        attn_entropy = 0.0

    attention_mask = inputs.get("attention_mask")
    seq_len = int(attention_mask.sum().item()) if attention_mask is not None else int(inputs["input_ids"].shape[-1])
    norm_pos = min(seq_len, MAX_PROMPT_TOKENS) / float(MAX_PROMPT_TOKENS)

    scalars = torch.tensor([entropy, prob_gap, logit_gap, attn_entropy, norm_pos], dtype=torch.float32)
    return torch.cat([hidden_last, scalars], dim=0)


def feature_with_temperature(base_features: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.cat([base_features, torch.tensor([float(tau)], dtype=torch.float32)], dim=0)


def save_checkpoint(mlp, optimizer, config: SavedConfig, step: int, epoch: int, best_loss: float) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mlp_state_dict": mlp.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "best_loss": best_loss,
            "config": asdict(config),
        },
        OUTPUT_DIR / "adaptive_temperature_mlp.pt",
    )
    atomic_write_json(OUTPUT_DIR / "config.json", asdict(config))


def maybe_load_checkpoint(mlp, optimizer, device: torch.device) -> tuple[int, int, float]:
    ckpt_path = OUTPUT_DIR / "adaptive_temperature_mlp.pt"
    if not ckpt_path.exists():
        return 0, 0, float("inf")
    ckpt = torch.load(ckpt_path, map_location=device)
    mlp.load_state_dict(ckpt["mlp_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"Resumed MLP checkpoint from {ckpt_path}")
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0)), float(ckpt.get("best_loss", float("inf")))

# ============================================================
# Main
# ============================================================

def main() -> None:
    set_seed(SEED)
    device = get_device()
    dtype = get_dtype()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME_OR_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"dtype: {dtype}")
    print(f"USE_ATTENTION_ENTROPY: {USE_ATTENTION_ENTROPY}")

    tokenizer, model = load_ministral3(device, dtype)
    hidden_size = get_hidden_size_from_config(model)
    base_dim = hidden_size + 5
    input_dim = base_dim + 1

    mlp = TemperaturePreferenceMLP(input_dim, MLP_HIDDEN_DIM, DROPOUT).to(device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    config = SavedConfig(
        model_name_or_path=MODEL_NAME_OR_PATH,
        input_dim=input_dim,
        hidden_size=hidden_size,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        dropout=DROPOUT,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        temperature_grid=TEMPERATURE_GRID,
        use_attention_entropy=USE_ATTENTION_ENTROPY,
        dtype_name=DTYPE_NAME,
        note="Per-sequence temperature scorer. One temperature is chosen before generation; this is not token-level adaptive decoding.",
    )

    global_step, start_epoch, best_loss = maybe_load_checkpoint(mlp, optimizer, device)

    rows = load_jsonl(DATASET_PATH)
    rows = [r for r in rows if "question" in r and "chosen_temperature" in r and "rejected_temperature" in r]
    if not rows:
        raise RuntimeError(f"No valid rows found in {DATASET_PATH}")

    print(f"Training rows: {len(rows)}")
    print("Objective: Bradley-Terry pair loss: score(prompt, chosen_tau) > score(prompt, rejected_tau)")
    print()

    start_time = time.perf_counter()
    mlp.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, EPOCHS):
        perm = torch.randperm(len(rows)).tolist()
        epoch_loss = 0.0
        epoch_count = 0

        for local_i, row_idx in enumerate(perm, start=1):
            row = rows[row_idx]
            question = row["question"]
            chosen_tau = float(row["chosen_temperature"])
            rejected_tau = float(row["rejected_temperature"])

            base = extract_prompt_signals(model, tokenizer, question, device)

            x_chosen = feature_with_temperature(base, chosen_tau).unsqueeze(0).to(device)
            x_rejected = feature_with_temperature(base, rejected_tau).unsqueeze(0).to(device)

            s_chosen = mlp(x_chosen)
            s_rejected = mlp(x_rejected)
            loss = F.softplus(-(s_chosen - s_rejected - MARGIN)).mean()
            (loss / GRAD_ACCUM_STEPS).backward()

            epoch_loss += float(loss.item())
            epoch_count += 1

            if local_i % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                avg_loss = epoch_loss / max(epoch_count, 1)
                best_loss = min(best_loss, avg_loss)

                if global_step % LOG_EVERY_STEPS == 0:
                    elapsed = time.perf_counter() - start_time
                    examples_done = epoch * len(rows) + local_i
                    total_examples = EPOCHS * len(rows)
                    pct = 100 * examples_done / total_examples
                    rate = examples_done / elapsed if elapsed > 0 else 0.0
                    remaining = total_examples - examples_done
                    eta = remaining / rate if rate > 0 else 0.0
                    print(
                        f"step={global_step} epoch={epoch+1}/{EPOCHS} "
                        f"row={local_i}/{len(rows)} ({pct:.2f}%) "
                        f"loss={avg_loss:.4f} best={best_loss:.4f} "
                        f"elapsed={fmt_time(elapsed)} eta={fmt_time(eta)}"
                    )

                if global_step % SAVE_EVERY_STEPS == 0:
                    save_checkpoint(mlp, optimizer, config, global_step, epoch, best_loss)

        if epoch_count % GRAD_ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        avg_epoch_loss = epoch_loss / max(epoch_count, 1)
        best_loss = min(best_loss, avg_epoch_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} done | avg_loss={avg_epoch_loss:.4f}")
        save_checkpoint(mlp, optimizer, config, global_step, epoch + 1, best_loss)
        if device.type == "mps":
            torch.mps.empty_cache()

    elapsed = time.perf_counter() - start_time
    summary = {
        "model": MODEL_NAME_OR_PATH,
        "dataset_path": DATASET_PATH,
        "rows": len(rows),
        "epochs": EPOCHS,
        "global_step": global_step,
        "best_loss": best_loss,
        "runtime_seconds": elapsed,
        "runtime_text": fmt_time(elapsed),
        "note": config.note,
    }
    atomic_write_json(OUTPUT_DIR / "train_summary.json", summary)
    print("\nDone.")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {OUTPUT_DIR / 'adaptive_temperature_mlp.pt'}")


if __name__ == "__main__":
    main()
