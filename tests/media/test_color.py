from __future__ import annotations

from dataclasses import replace

from engvit.media.color import classify_color
from tests.media.test_geometry import stream


def test_classify_color_rejects_pq_and_hlg() -> None:
    pq = replace(stream(width=1920, height=1080), color_transfer="smpte2084")
    hlg = replace(stream(width=1920, height=1080), color_transfer="arib-std-b67")
    assert classify_color(pq).state == "reject"
    assert classify_color(hlg).state == "reject"
    assert classify_color(pq).reason == "unsupported_hdr_transfer:smpte2084"


def test_classify_color_requires_choice_when_tags_are_missing() -> None:
    unknown = replace(
        stream(width=1920, height=1080),
        color_range=None,
        color_space=None,
        color_transfer=None,
        color_primaries=None,
    )
    assert classify_color(unknown).state == "needs_choice"


def test_classify_color_accepts_explicit_bt709_sdr() -> None:
    decision = classify_color(stream(width=1920, height=1080))
    assert decision.state == "allowed"
    assert decision.working_space == "linear-srgb"
    assert decision.output_color == {
        "color_range": "tv",
        "color_space": "bt709",
        "color_transfer": "bt709",
        "color_primaries": "bt709",
    }


def test_classify_color_rejects_dolby_vision_and_bt2020() -> None:
    dovi = replace(
        stream(width=1920, height=1080),
        side_data_types=("DOVI configuration record",),
    )
    wide = replace(
        stream(width=1920, height=1080),
        color_primaries="bt2020",
        color_space="bt2020nc",
    )
    assert classify_color(dovi).state == "reject"
    assert classify_color(dovi).reason == "unsupported_hdr_format:dolby_vision"
    assert classify_color(wide).state == "reject"
