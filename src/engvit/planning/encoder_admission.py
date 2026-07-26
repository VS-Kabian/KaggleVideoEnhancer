"""Evidence contract for target encoder admission."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EncoderProbe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    width: int = Field(ge=2)
    height: int = Field(ge=2)
    codec: str
    profile: str
    pixel_format: str
    frames: int = Field(ge=1)
    pts_contiguous: bool
    software_decode_passed: bool
    hardware_decode_passed: bool


class EncoderAdmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    admitted: bool
    reasons: tuple[str, ...]


def admit_encoder(
    probe: EncoderProbe,
    *,
    require_8k: bool,
) -> EncoderAdmission:
    """Require exact target evidence; 8K additionally requires Main10 and CPU decode."""
    reasons: list[str] = []
    if not probe.pts_contiguous:
        reasons.append("encoder PTS are not contiguous")
    if not probe.software_decode_passed:
        reasons.append("software decode did not pass")
    if require_8k:
        if (probe.width, probe.height) != (7680, 4320):
            reasons.append("8K probe geometry is not 7680x4320")
        if probe.codec != "hevc" or probe.profile.casefold() != "main 10":
            reasons.append("8K probe is not HEVC Main 10")
        if probe.pixel_format not in {"yuv420p10le", "p010le"}:
            reasons.append("8K probe did not preserve a 10-bit pixel format")
        if probe.frames < 2:
            reasons.append("8K probe requires at least two high-entropy frames")
    else:
        if probe.width > 3840 or probe.height > 2160:
            reasons.append("non-8K admission exceeds the 4K geometry bound")
        if probe.codec not in {"h264", "hevc"}:
            reasons.append("delivery codec is unsupported")
    return EncoderAdmission(admitted=not reasons, reasons=tuple(reasons))

