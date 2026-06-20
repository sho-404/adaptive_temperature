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

def _resolve_local_or_repo(name_or_path: str) -> str:
    """Prefer a fully-cached LOCAL snapshot so loading never hits the network.

    The MistralCommonBackend tokenizer otherwise issues a list_repo_files HF Hub
    request that HANGS when unauthenticated (and errors under offline mode). If
    the model is already cached we resolve to that local directory; only when
    nothing is cached do we fall back to the repo id (a real network download,
    which needs HF auth for this gated model).
    """
    if Path(name_or_path).exists():
        return name_or_path
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(name_or_path, local_files_only=True)
    except Exception:
        return name_or_path


def load_model(cfg: dict, device: torch.device, dtype: torch.dtype):
    source = _resolve_local_or_repo(cfg["model_name_or_path"])
    tokenizer = MistralCommonBackend.from_pretrained(source)

    kwargs = dict(dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)
    if cfg.get("use_attention_entropy"):
        kwargs["attn_implementation"] = "eager"  # match the adaptive evaluator's path

    try:
        model = Mistral3ForConditionalGeneration.from_pretrained(source, **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        model = Mistral3ForConditionalGeneration.from_pretrained(source, **kwargs)

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


def _eos_ids(tokenizer) -> list[int]:
    e = getattr(tokenizer, "eos_token_id", None)
    if e is None:
        return []
    return list(e) if isinstance(e, (list, tuple)) else [int(e)]


@torch.no_grad()
def generate_samples(model, tokenizer, messages, temperature, k, device, cfg, seed) -> list[tuple[str, bool]]:
    """Return k (response_text, truncated) tuples for a prompt at one temperature.

    `truncated` is True when the sample hit the max_new_tokens budget WITHOUT
    emitting an end-of-sequence token (i.e. it was cut off mid-generation, so a
    missing #### answer line is a truncation artifact, not a real failure).

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

    eos_ids = set(_eos_ids(tokenizer))
    results: list[tuple[str, bool]] = []
    for o in outputs:
        gen = o[input_len:]
        text = tokenizer.decode(gen, skip_special_tokens=True).strip()
        # Finished if an EOS appears in the generated tokens; else it was cut off
        # at the token budget. (If we can't resolve EOS, assume finished.)
        truncated = bool(eos_ids) and not (eos_ids & set(gen.tolist()))
        results.append((text, truncated))
    return results


# ============================================================
# vLLM backend (high-throughput; same model, batched generation)
# ============================================================
#
# vLLM changes only HOW samples are produced (many concurrent generations via
# continuous batching) — never WHICH samples or how pairs are selected. The
# driver still does greedy triage -> easy sweep -> hard round-robin race, and
# passes a list of (temperature, seed) "specs" per prompt; we run them as one
# batched call. Each spec maps 1:1 to one returned (text, truncated), in order.

def load_vllm(cfg: dict):
    from vllm import LLM

    source = _resolve_local_or_repo(cfg["model_name_or_path"])
    max_len = int(cfg["max_prompt_tokens"]) + int(cfg["max_new_tokens"])
    return LLM(
        model=source,
        tokenizer_mode="mistral",   # Ministral uses the tekken (mistral_common) tokenizer
        dtype=cfg["dtype_name"],
        max_model_len=max_len,
        gpu_memory_utilization=0.90,
        # Skip torch.compile / CUDA-graph capture. Avoids depending on a full
        # CUDA toolchain at startup; slightly less optimal decode but batching
        # dominates throughput. Generation correctness/logic is unaffected.
        enforce_eager=True,
    )


def vllm_generate(llm, cfg, messages, specs) -> list[tuple[str, bool]]:
    """Run one prompt at many (temperature, seed) specs in a single batched call.

    Returns a list of (text, truncated) aligned 1:1 with `specs`. truncated is
    True when vLLM stopped at the token budget (finish_reason == "length")
    rather than an end-of-sequence token — same meaning as the HF path.
    """
    from vllm import SamplingParams

    sampling = [
        SamplingParams(
            temperature=max(float(t), 0.0),  # 0.0 -> greedy (deterministic)
            top_p=1.0,
            max_tokens=int(cfg["max_new_tokens"]),
            seed=int(s),
            n=1,
        )
        for (t, s) in specs
    ]
    conversations = [messages for _ in specs]
    outputs = llm.chat(conversations, sampling, use_tqdm=False)

    results: list[tuple[str, bool]] = []
    for o in outputs:
        comp = o.outputs[0]
        results.append((comp.text.strip(), comp.finish_reason == "length"))
    return results
