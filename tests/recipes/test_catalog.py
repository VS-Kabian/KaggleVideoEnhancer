from __future__ import annotations

from engvit.recipes.catalog import ReleaseCapabilities


def test_release_capabilities_load_current_fail_closed_file() -> None:
    capabilities = ReleaseCapabilities.from_json_file("release-capabilities.json")
    assert capabilities.four_k is False
    assert capabilities.eight_k is False
    assert capabilities.rife is False

