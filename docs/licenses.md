# License and model policy

No neural model is executable merely because a repository or checkpoint is
popular.

An enabled model requires:

1. an allowlisted model ID and architecture signature;
2. a pinned upstream code archive/commit and SHA-256;
3. recorded code and weight terms permitting the intended use and
   redistribution;
4. an approved SafeTensors file with exact SHA-256 and tensor-layout checks;
5. loader parity against the official path on a locked corpus;
6. active Kaggle Python/Torch/CUDA compatibility;
7. full resource and quality acceptance receipts.

The included Real-ESRGAN-family registry, RIFE lock, and temporal locks are
disabled. Their URLs are research references, not license approval. PyTorch
pickle checkpoints are conversion inputs only in a quarantined, network-off
environment and are never accepted by the runtime loader.

FFmpeg licensing depends on the exact binary build configuration. EngVit records
the binary hash, version, license output, and build configuration in each
environment artifact; downstream distribution decisions must use that evidence.
