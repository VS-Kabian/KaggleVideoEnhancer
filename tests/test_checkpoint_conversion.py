from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_checkpoint_converter_requires_isolated_networkless_process(
    tmp_path: Path,
) -> None:
    """Catches loading a pickle checkpoint in an ordinary notebook process."""
    source = tmp_path / "legacy.pth"
    output = tmp_path / "converted.safetensors"
    source.write_bytes(b"not-loaded")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"ENGVIT_CONVERSION_ISOLATED", "ENGVIT_NETWORK_DISABLED"}
    }
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/convert_checkpoint.py",
            "--input",
            str(source),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 3
    assert "isolated, network-disabled" in completed.stderr
    assert not output.exists()

