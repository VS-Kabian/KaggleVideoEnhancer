"""Declarative FFmpeg fixtures used by media-contract integration tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FixtureSpec:
    fixture_id: str
    output_file: str
    args: tuple[str, ...]
    purpose: str


def _video_source(rate: str = "30", duration: str = "2") -> tuple[str, ...]:
    return (
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=160x90:rate={rate}:duration={duration}",
    )


def fixture_specs() -> tuple[FixtureSpec, ...]:
    """Return the stable, root-independent adversarial fixture contract."""
    return (
        FixtureSpec(
            "b_frames",
            "b-frames.mkv",
            (*_video_source(), "-c:v", "mpeg4", "-bf", "2", "-g", "12", "-q:v", "3", "{output}"),
            "B-frame decode order differs from presentation order.",
        ),
        FixtureSpec(
            "bt601_limited",
            "bt601-limited.mkv",
            (
                *_video_source(),
                "-vf",
                "setparams=range=limited:color_primaries=smpte170m:"
                "color_trc=smpte170m:colorspace=smpte170m",
                "-c:v",
                "ffv1",
                "{output}",
            ),
            "Limited-range SD color tags must not be treated as BT.709.",
        ),
        FixtureSpec(
            "bt709_full",
            "bt709-full.mkv",
            (
                *_video_source(),
                "-vf",
                "setparams=range=full:color_primaries=bt709:"
                "color_trc=bt709:colorspace=bt709",
                "-c:v",
                "ffv1",
                "{output}",
            ),
            "Full-range BT.709 must retain its explicit range.",
        ),
        FixtureSpec(
            "cfr_30000_1001",
            "cfr-30000-1001.mkv",
            (
                *_video_source("30000/1001"),
                "-c:v",
                "ffv1",
                "-r",
                "30000/1001",
                "-fps_mode",
                "cfr",
                "{output}",
            ),
            "NTSC-derived rational rates must remain exact.",
        ),
        FixtureSpec(
            "hlg",
            "hlg.mkv",
            (
                *_video_source(),
                "-vf",
                "format=yuv420p10le,setparams=range=limited:color_primaries=bt2020:"
                "color_trc=arib-std-b67:colorspace=bt2020nc",
                "-c:v",
                "ffv1",
                "{output}",
            ),
            "HLG input must fail the MVP HDR gate.",
        ),
        FixtureSpec(
            "interlaced",
            "interlaced-tff.mkv",
            (
                *_video_source(),
                "-vf",
                "tinterlace=mode=interleave_top",
                "-c:v",
                "mpeg2video",
                "-flags",
                "+ildct+ilme",
                "-top",
                "1",
                "{output}",
            ),
            "Field order and explicit bwdif behavior must be tested.",
        ),
        FixtureSpec(
            "irregular_vfr",
            "irregular-vfr.mkv",
            (
                *_video_source(),
                "-vf",
                "select='not(eq(mod(n,7),0))'",
                "-fps_mode",
                "vfr",
                "-c:v",
                "ffv1",
                "{output}",
            ),
            "Irregular timestamps must be explicitly normalized.",
        ),
        FixtureSpec(
            "malformed_probe",
            "malformed-probe.json",
            (),
            "Nullable and malformed FFprobe rationals must fail safely.",
        ),
        FixtureSpec(
            "multi_stream",
            "multi-stream.mkv",
            (
                *_video_source(),
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000:duration=2",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=880:sample_rate=48000:duration=2",
                "-f",
                "srt",
                "-i",
                "{subtitle}",
                "-f",
                "ffmetadata",
                "-i",
                "{chapters}",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-map",
                "2:a:0",
                "-map",
                "3:s:0",
                "-map_chapters",
                "4",
                "-c:v",
                "ffv1",
                "-c:a",
                "flac",
                "-c:s",
                "srt",
                "-metadata:s:a:0",
                "language=eng",
                "-metadata:s:a:1",
                "language=fra",
                "-disposition:a:0",
                "default",
                "-disposition:s:0",
                "forced",
                "-attach",
                "{attachment}",
                "-metadata:s:t:0",
                "mimetype=text/plain",
                "-metadata:s:t:0",
                "filename=fixture-attachment.txt",
                "{output}",
            ),
            "Ancillary stream mapping, offsets, dispositions, and attachments.",
        ),
        FixtureSpec(
            "negative_start_pts",
            "negative-start-pts.mkv",
            (
                *_video_source(),
                "-vf",
                "setpts=PTS-0.5/TB",
                "-copyts",
                "-avoid_negative_ts",
                "disabled",
                "-c:v",
                "ffv1",
                "{output}",
            ),
            "Negative presentation origins must be shifted consistently.",
        ),
        FixtureSpec(
            "non_square_sar",
            "non-square-sar.mkv",
            (*_video_source(), "-vf", "setsar=4/3", "-c:v", "ffv1", "{output}"),
            "Anamorphic input must become square-pixel exactly once.",
        ),
        FixtureSpec(
            "pq",
            "pq.mkv",
            (
                *_video_source(),
                "-vf",
                "format=yuv420p10le,setparams=range=limited:color_primaries=bt2020:"
                "color_trc=smpte2084:colorspace=bt2020nc",
                "-c:v",
                "ffv1",
                "{output}",
            ),
            "PQ input must fail the MVP HDR gate.",
        ),
        FixtureSpec(
            "rotation_flip",
            "rotation-flip.mkv",
            (
                *_video_source(),
                "-vf",
                "hflip",
                "-metadata:s:v:0",
                "rotate=90",
                "-c:v",
                "ffv1",
                "{output}",
            ),
            "Display transforms must not be applied twice.",
        ),
        FixtureSpec(
            "telecine_23",
            "telecine-23.mkv",
            (
                *_video_source("24000/1001"),
                "-vf",
                "telecine=pattern=23",
                "-c:v",
                "mpeg2video",
                "-flags",
                "+ildct+ilme",
                "{output}",
            ),
            "IVTC chunk boundaries must preserve cadence phase.",
        ),
    )
