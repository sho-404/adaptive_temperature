import time, torch
from huggingface_hub import snapshot_download
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

M = "mistralai/Ministral-3-3B-Instruct-2512-BF16"

local = snapshot_download(M, local_files_only=True)  # resolves cache, no download
print("local snapshot:", local, flush=True)

t = time.time(); print("tokenizer...", flush=True)
tok = MistralCommonBackend.from_pretrained(local)
print(f"  ok {time.time()-t:.1f}s", flush=True)

t = time.time(); print("model...", flush=True)
model = Mistral3ForConditionalGeneration.from_pretrained(
    local, dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="eager")
print(f"  ok {time.time()-t:.1f}s", flush=True)

t = time.time(); print("to mps...", flush=True)
model.to("mps")
print(f"  ok {time.time()-t:.1f}s DONE", flush=True)