from __future__ import annotations

from pathlib import Path

import pytest

from engvit.media.probe import parse_ffprobe
from engvit.media.selection import Selection
from engvit.types import Rational, VideoStreamInfo


def selection(tmp_path: Path, stream_index: int = 0) -> Selection:
    source = tmp_path / "source.mkv"
    source.write_bytes(b"source")
    return Selection(
        dataset_handle="owner/videos",
        dataset_version="1",
        relative_path="source.mkv",
        canonical_path=source.resolve(),
        bytes=6,
        source_sha256="a" * 64,
        video_stream_index=stream_index,
    )


def test_parse_ffprobe_preserves_all_streams_and_nullable_rationals(
    tmp_path: Path,
) -> None:
    payload = {
        "format": {"format_name": "matroska,webm", "duration": "2.500"},
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "ffv1",
                "width": 720,
                "height": 576,
                "sample_aspect_ratio": "16:15",
                "avg_frame_rate": "30000/1001",
                "r_frame_rate": "0/0",
                "time_base": "1/1000",
                "start_pts": -500,
                "duration_ts": 2500,
                "pix_fmt": "yuv420p10le",
                "bits_per_raw_sample": "10",
                "field_order": "tt",
                "color_range": "tv",
                "color_space": "bt709",
                "color_transfer": "bt709",
                "color_primaries": "bt709",
                "disposition": {"default": 1},
                "tags": {"language": "eng"},
                "side_data_list": [
                    {
                        "side_data_type": "Display Matrix",
                        "displaymatrix": (
                            "00000000:       65536           0           0\n"
                            "00000001:           0       65536           0\n"
                            "00000002:           0           0  1073741824"
                        ),
                    }
                ],
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "flac",
                "time_base": "1/1000",
                "start_pts": -250,
                "duration_ts": 2500,
                "disposition": {"default": 1},
                "tags": {"language": "fra", "title": "French"},
            },
        ],
    }
    info = parse_ffprobe(payload, selection(tmp_path))
    assert len(info.streams) == 2
    video = info.streams[0]
    assert isinstance(video, VideoStreamInfo)
    assert video.avg_frame_rate == Rational(30000, 1001)
    assert video.real_frame_rate is None
    assert video.sample_aspect_ratio == Rational(16, 15)
    assert video.pixel_format == "yuv420p10le"
    assert video.bits_per_raw_sample == 10
    assert video.display_matrix == (
        65536.0,
        0.0,
        0.0,
        0.0,
        65536.0,
        0.0,
        0.0,
        0.0,
        1073741824.0,
    )
    assert info.streams[1].metadata["title"] == "French"


def test_parse_ffprobe_rejects_missing_selected_video(tmp_path: Path) -> None:
    payload = {
        "format": {"format_name": "matroska"},
        "streams": [{"index": 0, "codec_type": "audio", "codec_name": "flac"}],
    }
    with pytest.raises(ValueError, match="selected video"):
        parse_ffprobe(payload, selection(tmp_path, stream_index=3))


def test_parse_ffprobe_rejects_encrypted_video(tmp_path: Path) -> None:
    payload = {
        "format": {"format_name": "matroska"},
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "side_data_list": [
                    {"side_data_type": "Encryption initialization data"}
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="encrypted"):
        parse_ffprobe(payload, selection(tmp_path))
