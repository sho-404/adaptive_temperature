"""ShroomCap task: load data and score hallucination detection.

Expected best at MID temperature. Source papers are git-ignored (see **/papers/).

Contract — every task module must expose these two functions:

    def load(lang):                 # lang is "en" or "es"
        '''Return the list of examples for this task/language.'''
        ...

    def score(outputs, data):       # outputs = model generations, data = load() result
        '''Return the metric for these generations.'''
        ...

The eval drivers (eval_fixed.py / eval_adaptive.py) only ever call these two.
"""
# TODO: port from legacy/evaluate_shroomcap.py
