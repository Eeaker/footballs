"""Canonical identity-resolution entry point.

The compatibility normalizer keeps legacy test/import environments safe when
multiple engine entry-point folders are placed on ``sys.path``.
"""
import os
import sys

sys.dont_write_bytecode = True
from scripts.run_pipeline import main


def normalize_thread_environment(environment=None):
    environment = os.environ if environment is None else environment
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        try:
            valid = int(environment.get(name, "1")) > 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            environment[name] = "1"
    return environment


if __name__ == "__main__":
    main()
