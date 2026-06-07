"""Combine the three task results into the 33/33/33 benchmark score.

Reads results/{gsm8k,humaneval,shroomcap}_<lang>.json and writes the headline
number (each task worth 1/3) to results/combined_<lang>.json

Usage:
    python combined_score.py --lang en
"""
# TODO: new file — aggregates the per-task results.
