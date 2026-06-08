"""Model backend for THIS container (Ministral / Mistral3), shared by the
drivers (eval_fixed.py, generate_pairs.py).

This is the only file that knows how to load and run the model. When you copy
this container for another model, this is the part you swap; the drivers and
the task modules stay the same.
"""

from __future__ import annotations

import json
from pathlib import Path

import logging

import torch
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
from transformers.utils import logging as hf_logging

HERE = Path(__file__).parent


# Mute ONLY the harmless, repeated "Both max_new_tokens and max_length seem to
# have been set" generation warning. All other transformers logging is kept.
class _DropMaxLengthWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not ("max_new_tokens" in msg and "max_length" in msg)


_hf_root = hf_logging.get_logger()  # configures + returns the 'transformers' logger
_hf_root.addFilter(_DropMaxLengthWarning())
for _h in _hf_root.handlers:        # child loggers propagate to these handlers
    _h.addFilter(_DropMaxLengthWarning())


# ============================================================
# Config / device / dtype
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
    if requested in ("auto", "cuda") and torch.cuda.is_available():
        device = torch.device("cuda")
    elif requested in ("auto", "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        if requested not in ("auto", "cpu"):
            print(f"WARNING: requested {requested!r} unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    announce_device(device, requested)
    return device


def announce_device(device: torch.device, requested: str) -> None:
    """Print a loud, unmistakable banner stating which processor is in use."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        body = [f">>> RUNNING ON: CUDA GPU  ({name}, {mem_gb:.0f} GB)",
                f">>> CUDA device count: {torch.cuda.device_count()}  |  torch CUDA {torch.version.cuda}"]
    elif device.type == "mps":
        body = [">>> RUNNING ON: APPLE GPU (MPS / Metal)"]
    else:
        body = [">>> RUNNING ON: CPU  <-- NO GPU DETECTED (this will be slow!)"]
    bar = "=" * 64
    print("\n" + bar)
    for line in body:
        print(line)
    print(f">>> (config requested device = {requested!r})")
    print(bar + "\n")


# ============================================================
# Model loading (specific to this container)
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
def generate_samples(model, tokenizer, messages, temperature, k, device, cfg, seed) -> list[str]:
    """Return k response strings for a prompt at one temperature.

    temperature == 0 -> greedy, which is deterministic, so it is generated ONCE
    regardless of k (sampling it k times would just copy the same text).

    For temperature > 0 the k samples come from a single batched generate() call
    via num_return_sequences=k: one manual_seed gives k genuinely different
    samples (sampling diverges across the batch) and is reproducible run-to-run.
    Pass a distinct seed per call so separate calls don't repeat the same RNG.
    """
    inputs = encode(tokenizer, messages, device, cfg["max_prompt_tokens"])
    input_len = inputs["input_ids"].shape[-1]
    do_sample = float(temperature) > 0.0
    n = int(k) if do_sample else 1

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=cfg["max_new_tokens"],
        do_sample=do_sample,
        use_cache=True,
        num_return_sequences=n,
    )
    if do_sample:
        gen_kwargs["temperature"] = max(float(temperature), 1e-5)
        gen_kwargs["top_p"] = 1.0

    torch.manual_seed(int(seed))
    outputs = model.generate(**gen_kwargs)
    return [tokenizer.decode(o[input_len:], skip_special_tokens=True).strip() for o in outputs]
