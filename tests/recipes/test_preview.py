from __future__ import annotations

from pathlib import Path

from engvit.recipes.preview import PreviewItem, render_preview_html


def test_preview_is_accessible_escaped_and_path_private(tmp_path: Path) -> None:
    private = tmp_path / "private-name.mp4"
    html = render_preview_html(
        (
            PreviewItem(
                candidate_id="candidate-a",
                label="<Detail & sharp>",
                video_filename="candidate-a.mp4",
                crop_filenames=("candidate-a-01.png",),
                source_path=private,
            ),
        )
    )
    assert "<Detail & sharp>" not in html
    assert "&lt;Detail &amp; sharp&gt;" in html
    assert "autoplay" not in html
    assert str(private) not in html
    assert 'controls' in html
    assert 'aria-label=' in html

