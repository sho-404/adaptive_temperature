"""Train the adaptive-temperature MLP for ONE language.

Reads datasets/*_<lang>.jsonl (all tasks fused, or a subset) and writes the
trained controller to checkpoints/mlp_<lang>.pt

Usage:
    python train_mlp.py --lang en
"""
# TODO: port from legacy/train_adaptive_decoder.py
