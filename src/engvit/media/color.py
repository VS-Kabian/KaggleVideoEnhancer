"""Fail-closed color classification for the SDR-only first release."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from engvit.types import VideoStreamInfo

_HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})
_SDR_TRANSFERS = frozenset(
    {"bt709", "smpte170m", "gamma22", "gamma28", "iec61966-2-1"}
)
_SDR_SPACES = frozenset({"bt709", "smpte170m", "bt470bg"})
_SDR_PRIMARIES = frozenset({"bt709", "smpte170m", "bt470bg"})


class ColorDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: Literal["allowed", "reject", "needs_choice"]
    reason: str | None
    working_space: str | None
    output_color: dict[str, str] | None


def classify_color(video: VideoStreamInfo) -> ColorDecision:
    """Classify explicit SDR metadata and reject known HDR transfer functions."""
    transfer = video.color_transfer
    if any(
        "dovi" in item.casefold() or "dolby vision" in item.casefold()
        for item in video.side_data_types
    ):
        return ColorDecision(
            state="reject",
            reason="unsupported_hdr_format:dolby_vision",
            working_space=None,
            output_color=None,
        )
    if transfer in _HDR_TRANSFERS:
        return ColorDecision(
            state="reject",
            reason=f"unsupported_hdr_transfer:{transfer}",
            working_space=None,
            output_color=None,
        )
    if video.color_primaries == "bt2020" or video.color_space in {
        "bt2020nc",
        "bt2020c",
    }:
        return ColorDecision(
            state="reject",
            reason="unsupported_hdr_or_wide_gamut:bt2020",
            working_space=None,
            output_color=None,
        )
    values = (
        video.color_range,
        video.color_space,
        video.color_transfer,
        video.color_primaries,
    )
    if any(value is None or value in {"unknown", "unspecified", "reserved"} for value in values):
        return ColorDecision(
            state="needs_choice",
            reason="incomplete_or_ambiguous_color_metadata",
            working_space=None,
            output_color=None,
        )
    if (
        video.color_range not in {"tv", "pc"}
        or video.color_space not in _SDR_SPACES
        or transfer not in _SDR_TRANSFERS
        or video.color_primaries not in _SDR_PRIMARIES
    ):
        return ColorDecision(
            state="needs_choice",
            reason="unsupported_or_ambiguous_color_metadata",
            working_space=None,
            output_color=None,
        )
    return ColorDecision(
        state="allowed",
        reason=None,
        working_space="linear-srgb",
        output_color={
            "color_range": video.color_range,
            "color_space": "bt709",
            "color_transfer": "bt709",
            "color_primaries": "bt709",
        },
    )
