from __future__ import annotations

import numpy as np

from engvit.analysis.features import extract_features


def test_edges_score_above_flat_image_without_fabricating_face_score() -> None:
    flat = np.full((64, 64, 3), 128, dtype=np.uint8)
    checker = np.indices((64, 64)).sum(axis=0) % 2 * 255
    textured = np.repeat(checker[:, :, None], 3, axis=2).astype(np.uint8)
    flat_features = extract_features(flat, None)
    edge_features = extract_features(textured, flat)
    assert edge_features.spatial_information > flat_features.spatial_information
    assert edge_features.edge_density > flat_features.edge_density
    assert flat_features.flat_fraction > edge_features.flat_fraction
    assert edge_features.face_score is None


def test_invalid_rgb_shape_is_rejected() -> None:
    invalid = np.zeros((32, 32), dtype=np.uint8)
    try:
        extract_features(invalid, None)
    except ValueError as exc:
        assert "RGB" in str(exc)
    else:
        raise AssertionError("non-RGB input must fail")

