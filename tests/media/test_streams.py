from __future__ import annotations

from engvit.media.streams import plan_ancillary_streams
from tests.media.pipeline_helpers import unused_stream


def test_mp4_policy_accounts_for_audio_and_omits_private_metadata() -> None:
    plan = plan_ancillary_streams((unused_stream(),), "mp4_compatibility")
    assert plan.receipts[0].action == "transcode"
    assert plan.receipts[0].target_codec == "aac"
    assert all("private canary" not in item for item in plan.output_options)
    assert "-map_metadata" in plan.output_options
    assert "-1" in plan.output_options


def test_mkv_policy_copies_compatible_ancillary_stream() -> None:
    plan = plan_ancillary_streams((unused_stream(),), "mkv_preservation")
    assert plan.receipts[0].action == "copy"

