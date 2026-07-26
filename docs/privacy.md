# Privacy

EngVit assumes user videos are sensitive.

- Kaggle Notebook visibility and every attached media/weight/wheel/
  continuation Dataset must be private.
- Internet must be disabled before source frames are read.
- Unknown visibility fails unless the operator records an explicit manual
  attestation after checking the Kaggle UI.
- Discovery follows only regular files inside declared roots; symlinks and path
  traversal are rejected.
- Source identity uses a complete SHA-256 while checking size/device/inode
  stability.
- Runtime output avoids source paths. Preview HTML uses generated basenames and
  escapes labels.
- Continuation ZIP extraction rejects absolute paths, `..`, symbolic links,
  unknown categories, unexpected members, size/hash drift, and non-empty
  destinations.
- Metadata is stripped from generated video segments and previews.

EngVit does not currently automate Kaggle Dataset visibility queries or private
Dataset publication. The operator must verify privacy before upload and after
reattach.
