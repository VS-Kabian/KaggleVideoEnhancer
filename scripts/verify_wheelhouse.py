"""Verify an EngVit wheelhouse and emit a machine-readable result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engvit.supply_chain import verify_wheelhouse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args(argv)
    report = verify_wheelhouse(args.root, args.lock)
    print(json.dumps(report.model_dump(mode="json"), sort_keys=True))
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

