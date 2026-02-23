"""
Entry point for ``python -m src``.

Delegates to the unified CLI in ``main.py``.

Usage
-----
::

    python -m src --mode generate --config configs/quick.yaml
    python -m src --mode train    --config configs/quick.yaml
    python -m src --mode evaluate --config configs/quick.yaml
    python -m src experiment --list
    python -m src pipeline --preset quick

For the full CLI help::

    python -m src --help
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    # Ensure the project root is on sys.path so ``main.py`` imports work
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from main import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
