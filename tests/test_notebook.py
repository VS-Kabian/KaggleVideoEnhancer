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
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"notebook-cell-{index}", "exec")
