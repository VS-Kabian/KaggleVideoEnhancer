"""Deterministic stratified preview/evaluation sample selection."""

from __future__ import annotations

import hashlib

from engvit.analysis.proxy_scan import ProxyRow, ProxyScan
from engvit.analysis.scenes import Scene

_EXTREME_FEATURES = (
    "spatial_information",
    "temporal_information",
    "noise_score",
    "blocking_score",
    "ringing_score",
    "banding_score",
    "motion",
    "dark_fraction",
    "highlight_fraction",
    "text_line_score",
)


def _maximum_row(rows: tuple[ProxyRow, ...], field: str) -> ProxyRow:
    return max(rows, key=lambda row: (float(getattr(row, field)), -row.output_index))


def _rank(source_sha256: str, index: int) -> bytes:
    return hashlib.sha256(f"{source_sha256}:{index}".encode()).digest()


def select_samples(
    scan: ProxyScan,
    scenes: tuple[Scene, ...],
    source_sha256: str,
    *,
    count: int,
) -> tuple[int, ...]:
    """Cover structural strata first, then fill by time and stable hash rank."""
    if len(source_sha256) != 64:
        raise ValueError("source_sha256 must contain 64 hexadecimal characters")
    try:
        bytes.fromhex(source_sha256)
    except ValueError as exc:
        raise ValueError("source_sha256 must be hexadecimal") from exc
    if count <= 0 or count > scan.output_frame_count:
        raise ValueError("sample count must fit the output timeline")
    if not scan.rows or not scenes:
        raise ValueError("sampling requires proxy rows and scenes")

    mandatory = {0, scan.output_frame_count - 1}
    for scene in scenes:
        mandatory.add((scene.start_frame + scene.end_frame - 1) // 2)
        if scene.start_frame > 0:
            mandatory.update((scene.start_frame - 1, scene.start_frame))
    for field in _EXTREME_FEATURES:
        mandatory.add(_maximum_row(scan.rows, field).output_index)
    if len(mandatory) > count:
        raise ValueError(
            f"sample count {count} cannot cover {len(mandatory)} mandatory strata"
        )

    selected = set(mandatory)
    remaining = count - len(selected)
    if remaining:
        for bin_index in range(remaining):
            target = int(
                (bin_index + 0.5)
                * scan.output_frame_count
                / remaining
            )
            candidates = [
                index
                for index in range(scan.output_frame_count)
                if index not in selected
            ]
            if not candidates:
                break
            choice = min(
                candidates,
                key=lambda index: (
                    abs(index - target),
                    _rank(source_sha256, index),
                ),
            )
            selected.add(choice)
    if len(selected) < count:
        candidates = sorted(
            (
                index
                for index in range(scan.output_frame_count)
                if index not in selected
            ),
            key=lambda index: _rank(source_sha256, index),
        )
        selected.update(candidates[: count - len(selected)])
    return tuple(sorted(selected))

