"""Run EngVit release acceptance aggregation from immutable case receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engvit.acceptance import (
    load_acceptance_matrix,
    release_capability_payload,
    run_acceptance,
)
from engvit.canonical import canonical_bytes
from engvit.orchestration.atomic import AtomicArtifactWriter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--capabilities-output", type=Path)
    arguments = parser.parse_args()

    matrix = load_acceptance_matrix(arguments.matrix)
    report = run_acceptance(matrix, evidence_root=arguments.evidence_root)
    writer = AtomicArtifactWriter()
    writer.write(
        arguments.report_output,
        canonical_bytes(report, projection="full"),
    )
    if arguments.capabilities_output is not None:
        writer.write(
            arguments.capabilities_output,
            json.dumps(
                release_capability_payload(report),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8"),
        )
    enabled = [name for name, value in report.capabilities.items() if value]
    print(
        json.dumps(
            {
                "report_sha256": report.report_sha256,
                "enabled_capabilities": enabled,
                "report": arguments.report_output.name,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
