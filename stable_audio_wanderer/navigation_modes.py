"""Canonical navigation names, with the pre-Wander mode accepted on input."""


def normalize_mode(value):
    mode = str(value).strip().lower()
    return 'wander' if mode == 'random' else mode
