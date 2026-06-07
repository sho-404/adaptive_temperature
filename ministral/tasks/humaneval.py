"""HumanEval task: load problems and score via pass@k.

Expected best at HIGH temperature.

Contract — every task module must expose these two functions:

    def load(lang):                 # lang is "en" or "es"
        '''Return the list of examples for this task/language.'''
        ...

    def score(outputs, data):       # outputs = model generations, data = load() result
        '''Return the metric (e.g. pass@k) for these generations.'''
        ...

The eval drivers (eval_fixed.py / eval_adaptive.py) only ever call these two.
"""
# TODO: port from legacy/evaluate_HumanEval.py
