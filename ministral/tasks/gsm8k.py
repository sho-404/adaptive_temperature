"""GSM8K task — all GSM8K-specific logic, backend-agnostic.

Expected best at LOW temperature.

This module is the single home for everything that is specific to GSM8K. The
drivers (eval_fixed.py / generate_pairs.py) own the model/backend and the loops;
they call into the functions below for the task-specific parts.

Task interface used by the drivers:

    load(lang, split)           -> list of examples [{index, question, answer_raw, ground_truth}]
    build_messages(question)    -> chat messages (for the HuggingFace chat template)
    build_prompt(question)      -> plain prompt string (for Ollama)
    extract_answer(response)    -> (Decimal | None, status_str)
    is_correct(parsed, gt)      -> bool

`ground_truth` and `parsed` are Decimal (or None). Drivers stringify for IO.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, getcontext

from datasets import load_dataset

getcontext().prec = 50

NAME = "gsm8k"
BEST_TEMPERATURE_HINT = "low"

ABS_TOL = Decimal("1e-6")

# When False, only explicit final-answer markers / boxed answers are accepted.
# Keep False: it gives safer labels (an unparseable generation is excluded
# rather than guessed from the last number in the text).
ALLOW_LAST_NUMBER_FALLBACK = False

# GSM8K source per language. English is openai/gsm8k. A Spanish source must be
# added here before `--lang es` will work (e.g. a translated GSM8K).
DATASETS = {
    "en": {"name": "openai/gsm8k", "config": "main"},
}


# ============================================================
# Dataset loading
# ============================================================

def load(lang: str = "en", split: str = "test") -> list[dict]:
    """Return GSM8K examples for a language/split.

    eval_fixed.py uses split="test"; generate_pairs.py uses split="train".
    """
    if lang not in DATASETS:
        raise NotImplementedError(
            f"No GSM8K source configured for lang={lang!r}. "
            f"Add one to DATASETS in tasks/gsm8k.py "
            f"(English is the only source wired up so far)."
        )

    info = DATASETS[lang]
    ds = load_dataset(info["name"], info["config"], split=split)

    examples: list[dict] = []
    for i, row in enumerate(ds):
        examples.append(
            {
                "index": i,
                "question": row["question"],
                "answer_raw": row["answer"],
                "ground_truth": extract_ground_truth(row["answer"]),
            }
        )
    return examples


# ============================================================
# Prompts (two shapes: chat messages for HF, plain text for Ollama)
# ============================================================

_RULES = """Solve the following grade-school math problem step by step. Take as much room as you need to reason.

Your answer is ONLY accepted if its VERY LAST line is exactly:
#### N
where N is the final numeric answer and nothing else: digits only, no words, no units, no symbols, no extra text.
For example, if the final answer is 42, the last line must be exactly:
#### 42

Strict rules:
1. Use #### ONLY on that final line. Never write #### anywhere else in your response.
2. Write nothing at all after that final #### line.
3. If the answer is not in this exact form, it is marked WRONG even if the reasoning is correct. We do not search the rest of the text for a number.

Problem:
{question}"""


def build_messages(question: str) -> list[dict[str, str]]:
    system = (
        "You are a careful grade-school math solver. You always finish with the "
        "final answer on its own last line as '#### N' (a single number, digits "
        "only, no units), and you never write #### anywhere else in the response."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _RULES.format(question=question)},
    ]


def build_prompt(question: str) -> str:
    return _RULES.format(question=question)


# ============================================================
# Numeric parsing / scoring
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


def parse_marker_value(text: str) -> Decimal | None:
    """Parse the number from an explicit final-answer marker line.

    Gated on a #### / "final answer:" marker, so it is safe to tolerate trailing
    units or words the model added against instructions (e.g. "72 clips"): try a
    strict parse first, then fall back to the last number on the line. This stops
    a correct answer being marked wrong purely over formatting (a false negative
    that would otherwise pollute the preference pairs).
    """
    value = parse_decimal(text)
    if value is not None:
        return value
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text.replace("$", ""))
    if nums:
        return parse_decimal(nums[-1])
    return None


def extract_ground_truth(answer: str) -> Decimal | None:
    match = re.search(r"####\s*([^\n]+)", answer)
    if not match:
        return None
    return parse_decimal(match.group(1))


def extract_answer(response: str) -> tuple[Decimal | None, str]:
    """Prefer explicit final markers, then boxed answers. Returns (value, status)."""
    # NOTE: do NOT anchor the marker to the start of the line. Models often emit
    # the final answer as a numbered list item, e.g. "5. #### 72", and a
    # ^-anchored pattern would miss it and wrongly count a correct answer as
    # unparseable. Matching the marker anywhere on the line (then taking the last
    # match) mirrors the lenient extract_ground_truth above.
    marker_patterns = [
        r"(?m)####\s*([^\n]+?)\s*$",
        r"(?im)final answer\s*[:=]\s*([^\n]+?)\s*$",
        r"(?im)answer\s*[:=]\s*([^\n]+?)\s*$",
    ]

    for pattern in marker_patterns:
        matches = re.findall(pattern, response)
        if matches:
            value = parse_marker_value(matches[-1])
            if value is not None:
                return value, "explicit_marker"

    boxed = re.findall(r"\\boxed\{([^{}]+)\}", response)
    if boxed:
        value = parse_marker_value(boxed[-1])
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


def is_correct(parsed: Decimal | None, ground_truth: Decimal | None) -> bool:
    if parsed is None or ground_truth is None:
        return False
    return abs(parsed - ground_truth) <= ABS_TOL
