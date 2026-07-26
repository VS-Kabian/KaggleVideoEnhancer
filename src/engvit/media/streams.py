"""Explicit ancillary-stream copy/transcode/omit policy."""

from __future__ import annotations

from dataclasses import dataclass

from engvit.types import ContainerPolicy, StreamInfo


@dataclass(frozen=True)
class StreamReceipt:
    source_index: int
    codec_type: str
    source_codec: str | None
    action: str
    target_codec: str | None
    reason: str


@dataclass(frozen=True)
class AncillaryStreamPlan:
    policy: ContainerPolicy
    output_options: tuple[str, ...]
    receipts: tuple[StreamReceipt, ...]


def plan_ancillary_streams(
    streams: tuple[StreamInfo, ...],
    policy: ContainerPolicy,
) -> AncillaryStreamPlan:
    """Account for every non-video stream without copying arbitrary metadata."""
    options: list[str] = ["-map_metadata", "-1"]
    receipts: list[StreamReceipt] = []
    type_counts = {"audio": 0, "subtitle": 0, "data": 0, "attachment": 0}
    for stream in streams:
        if stream.codec_type == "video":
            continue
        action = "omit"
        target: str | None = None
        reason = "unsupported_stream_type"
        if policy == "mkv_preservation" and stream.codec_type in type_counts:
            action = "copy"
            target = stream.codec_name
            reason = "mkv_preservation_copy"
        elif stream.codec_type == "audio":
            compatible = stream.codec_name in {
                "aac",
                "mp3",
                "ac3",
                "eac3",
                "alac",
            }
            action = "copy" if compatible else "transcode"
            target = stream.codec_name if action == "copy" else "aac"
            reason = "mp4_audio_compatible" if action == "copy" else "mp4_audio_transcode"
        elif stream.codec_type == "subtitle" and stream.codec_name in {
            "subrip",
            "srt",
            "ass",
            "webvtt",
            "mov_text",
        }:
            action = "copy" if stream.codec_name == "mov_text" else "transcode"
            target = "mov_text"
            reason = "mp4_text_subtitle"

        if action != "omit":
            options.extend(("-map", f"1:{stream.index}"))
            type_key = stream.codec_type
            output_index = type_counts[type_key]
            type_counts[type_key] += 1
            specifier = {
                "audio": "a",
                "subtitle": "s",
                "data": "d",
                "attachment": "t",
            }[type_key]
            options.extend((f"-c:{specifier}:{output_index}", target or "copy"))
            if stream.language is not None:
                options.extend(
                    (
                        f"-metadata:s:{specifier}:{output_index}",
                        f"language={stream.language}",
                    )
                )
            dispositions = tuple(
                name
                for name, enabled in sorted(stream.disposition.items())
                if enabled
            )
            if dispositions:
                options.extend(
                    (
                        f"-disposition:{specifier}:{output_index}",
                        "+".join(dispositions),
                    )
                )
        receipts.append(
            StreamReceipt(
                source_index=stream.index,
                codec_type=stream.codec_type,
                source_codec=stream.codec_name,
                action=action,
                target_codec=target,
                reason=reason,
            )
        )
    return AncillaryStreamPlan(
        policy=policy,
        output_options=tuple(options),
        receipts=tuple(receipts),
    )
