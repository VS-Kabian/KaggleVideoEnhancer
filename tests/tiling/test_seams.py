from __future__ import annotations

import numpy as np

from engvit.tiling.layout import plan_layout
from engvit.tiling.seams import analyze_seams


def test_seam_evidence_reports_zero_difference_for_reference_match() -> None:
    layout = plan_layout(
        width=61,
        height=47,
        tile_size=24,
        overlap=7,
        context=4,
    )
    image = np.zeros((94, 122, 3), dtype=np.uint8)
    evidence = analyze_seams(image, image.copy(), layout, scale=2)
    assert evidence.boundary_p99 == 0.0
    assert evidence.interior_p99 == 0.0
    assert evidence.zero_weight_pixels == 0

