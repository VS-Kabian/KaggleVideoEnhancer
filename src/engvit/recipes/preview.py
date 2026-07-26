"""Small accessible preview page construction without source-path leakage."""

from __future__ import annotations

import html
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_SAFE_FILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class PreviewItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    label: str
    video_filename: str
    crop_filenames: tuple[str, ...]
    source_path: Path


def _safe_name(value: str) -> str:
    if _SAFE_FILE.fullmatch(value) is None or Path(value).name != value:
        raise ValueError("preview filenames must be generated safe basenames")
    return value


def render_preview_html(items: tuple[PreviewItem, ...]) -> str:
    """Render keyboard/browser-native controls and generated public filenames."""
    sections: list[str] = []
    for item in items:
        if _SAFE_ID.fullmatch(item.candidate_id) is None:
            raise ValueError("preview candidate ID is unsafe")
        label = html.escape(item.label, quote=True)
        video = html.escape(_safe_name(item.video_filename), quote=True)
        images = "".join(
            (
                f'<img src="{html.escape(_safe_name(filename), quote=True)}" '
                f'alt="{label} lossless comparison crop">'
            )
            for filename in item.crop_filenames
        )
        sections.append(
            f'<section id="{item.candidate_id}" aria-label="{label}">'
            f"<h2>{label}</h2>"
            f'<video controls preload="metadata" aria-label="{label} video">'
            f'<source src="{video}" type="video/mp4">'
            "Your browser cannot play this preview."
            "</video>"
            f'<div class="crops">{images}</div>'
            "</section>"
        )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>EngVit preview comparison</title></head><body>"
        "<h1>EngVit preview comparison</h1>"
        + "".join(sections)
        + "</body></html>"
    )

