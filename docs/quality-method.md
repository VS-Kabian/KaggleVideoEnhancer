# Quality method

EngVit separates structural correctness from perceptual improvement.

## Per-job structural evidence

The current executable gate checks:

- exact decoded frame count;
- contiguous logical PTS from zero;
- exact output dimensions and time base;
- square-pixel sample aspect ratio;
- complete first/last boundary hashes.

Any failure makes the job fail. This evidence can show that chunking and concat
did not truncate, overlap, or reorder video; it cannot show that invented detail
is faithful.

## Release evidence

All perceptual thresholds begin `UNSET`. Neural release requires locked
synthetic-HR fidelity, encoder roundtrip, temporal/flicker/freeze, color,
banding, tile-seam, and blinded human protocols. Real-source blind metrics are
diagnostic and cannot substitute for an HR reference.

Formal human claims require the documented P.910/BT.500-style protocol and at
least 24 retained screened observers. Smaller reviews are labeled pilot or
informal.

`acceptance/matrix.yaml` names every required receipt. A capability becomes true
only if every referenced receipt is a strict `PASS`, contains environment/job/
artifact hashes, passes every named metric, and records independent audit.
Missing evidence remains `NOT_EVALUATED`.
