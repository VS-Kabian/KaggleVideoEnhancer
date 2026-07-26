"""Execute and persist a notebook with nbclient for CI/local validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    args = parser.parse_args()

    notebook = nbformat.read(args.input, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name=args.kernel,
        resources={"metadata": {"path": str(args.cwd.resolve())}},
    )
    try:
        client.execute()
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, args.output)
    nbformat.validate(notebook)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
