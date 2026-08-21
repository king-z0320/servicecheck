"""Keep the public command as ``python -m qc.eval``."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
