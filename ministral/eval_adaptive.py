"""Evaluate the MLP-driven adaptive temperature for ONE task and ONE language.

Loads checkpoints/mlp_<lang>.pt, picks the temperature per sequence, runs the
task, and writes the score to results/<task>_<lang>.json (under an "adaptive" key).

Only the requested --task is imported (import tasks/<task>.py lazily, inside
main), so an unfinished task module can never crash an unrelated run.

Usage:
    python eval_adaptive.py --task gsm8k --lang en
"""
# TODO: port from legacy/evaluate_adaptive_decoder.py
