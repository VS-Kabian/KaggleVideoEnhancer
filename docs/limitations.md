# Known limitations

Current executable scope:

- deterministic Lanczos spatial resize;
- SDR sources with explicit supported color metadata;
- sources up to 15 minutes;
- output up to 3840x2160;
- video-only H.264 Matroska delivery;
- single-worker chunk execution;
- structural video QA and private continuation ZIP creation.

Not yet implemented or qualified:

- enabled Real-ESRGAN inference with approved production weights;
- automatic quality-based recipe selection;
- full-duration proxy feature extraction in the notebook UI;
- target-geometry time/disk/VRAM admission in the notebook;
- audio, subtitle, chapter, attachment, and metadata remux in the Kaggle path;
- MP4 delivery;
- telecine IVTC mezzanine execution;
- HDR/Dolby Vision/BT.2020 processing or tone mapping;
- RIFE inference;
- temporal VSR inference;
- perceptual threshold calibration and formal human evidence;
- private Kaggle Dataset publishing/visibility API integration;
- full 10–15 minute minimum-GPU acceptance;
- 8K execution.

Therefore the current smoke and unit suites do not establish “production
ready,” “4K enhanced,” or “8K supported.” They establish a verified Phase 0
plumbing baseline and fail-closed extension points.
