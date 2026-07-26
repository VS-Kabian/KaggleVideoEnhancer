from __future__ import annotations

from engvit.planning.encoder_admission import EncoderProbe, admit_encoder


def test_8k_encoder_requires_exact_main10_geometry_and_software_decode() -> None:
    failed = admit_encoder(
        EncoderProbe(
            width=7680,
            height=4320,
            codec="hevc",
            profile="Main 10",
            pixel_format="yuv420p10le",
            frames=2,
            pts_contiguous=True,
            software_decode_passed=False,
            hardware_decode_passed=False,
        ),
        require_8k=True,
    )
    assert failed.admitted is False
    assert "software decode" in " ".join(failed.reasons)


def test_4k_h264_smoke_accepts_exact_geometry_and_pts() -> None:
    decision = admit_encoder(
        EncoderProbe(
            width=3840,
            height=2160,
            codec="h264",
            profile="High",
            pixel_format="yuv420p",
            frames=60,
            pts_contiguous=True,
            software_decode_passed=True,
            hardware_decode_passed=False,
        ),
        require_8k=False,
    )
    assert decision.admitted is True
