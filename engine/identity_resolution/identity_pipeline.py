"""Canonical identity-resolution entry point."""
import sys

sys.dont_write_bytecode = True
from scripts.run_pipeline import main


if __name__ == "__main__":
    main()
