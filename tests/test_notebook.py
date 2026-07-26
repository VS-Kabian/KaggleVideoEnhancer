from __future__ import annotations

from pathlib import Path

import nbformat


def test_kaggle_notebook_is_valid_and_has_ordered_product_sections() -> None:
    path = Path("notebooks/engvit_kaggle.ipynb")
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    positions = tuple(markdown.index(f"## {index}.") for index in range(1, 9))
    assert positions == tuple(sorted(positions))
    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    assert "SMOKE_MODE = True" in code
    assert "RUN_JOB = SMOKE_MODE" in code
    assert "SensitiveMediaPreflight().run(" in code
    assert "run_structural_qa(" in code
    cell_ids = tuple(cell.id for cell in notebook.cells)
    assert len(cell_ids) == len(set(cell_ids))
    assert all(cell_id.startswith("engvit-") for cell_id in cell_ids)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"notebook-cell-{index}", "exec")


def test_personalized_kaggle_notebook_has_exact_attached_dataset_paths() -> None:
    path = Path("notebooks/engvit_kaggle_mukikabi006.ipynb")
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    assert notebook.metadata["engvit_profile"] == "mukikabi006"
    assert (
        "/kaggle/input/datasets/mukikabi006/engvit-code"
        in code
    )
    assert (
        "/kaggle/input/datasets/mukikabi006/private-video-dataset"
        in code
    )
    assert 'RELATIVE_VIDEO_PATH = "GH011828.realesrgan.mkv"' in code
    assert "SMOKE_MODE = False" in code
    assert "RUN_JOB = True" in code
    assert "MAX_NEW_CHUNKS = 1" in code
    assert '("*/src/engvit", "*/*/src/engvit")' in code
    assert "attached_root.glob(pattern)" in code
    assert "ambiguous EngVit source roots" in code
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"personalized-notebook-cell-{index}", "exec")
