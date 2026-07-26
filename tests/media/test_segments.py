from __future__ import annotations

from engvit.media.segments import parse_framehash
from engvit.types import Rational


def test_parse_framehash_preserves_exact_time_base_pts_and_hashes() -> None:
    output = """#format: frame checksums
#tb 0: 1001/30000
0, 0, 0, 1, 10, aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
0, 1, 1, 1, 10, bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
"""
    scan = parse_framehash(output)
    assert scan.time_base == Rational(1001, 30000)
    assert tuple(frame.pts for frame in scan.frames) == (0, 1)

