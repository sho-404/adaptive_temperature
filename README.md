# adaptive_temperature

When can adaptive temperature work? An analysis of where and when correctness
becomes readable in LLM reasoning states (Ministral-3B, GSM8K).

Reference: *Adaptive Decoding via Latent Preference Optimization*
(Dhuliawala et al., 2024) — `adaptivedecoder.pdf`.

## Layout

```
paper/                        the writeup: draft, figures, tables
data/                         gsm8k pairs (jsonl) + cached features (.pt) — gitignored
scripts/
  model.py, config.json       shared model backend (HF, Ministral-3B)
  tasks/gsm8k.py              prompt template, answer parsing, scoring
  decoding/                   the adaptive-temperature attempt (results 1 & 3)
    generate_pairs.py           LPO preference pairs via temperature sweeps   [VM]
    extract_features.py         per-prompt features at final prompt token     [VM]
    train_mlp.py                conditional Bradley-Terry temperature scorer  [Mac]
  explainability/             the analysis: why it can't work (result 2 +)
    extract_reasoning_features.py  hidden states along reasoning (25-100%)    [VM]
    probe_reasoning.py             correctness probes, confound-controlled    [Mac]
    probe_positional.py            when the signal appears                    [Mac]
```

`[VM]` = needs the A100 box (GPU forward passes). `[Mac]` = runs locally off
the cached tensors in `data/`.

## Pipeline order

1. `decoding/generate_pairs.py` → `data/gsm8k_en.jsonl`
2. `decoding/extract_features.py` → `data/gsm8k_en.pt`
3. `explainability/extract_reasoning_features.py` → `data/gsm8k_en_reasoning.pt`
4. everything else runs locally off those three files
