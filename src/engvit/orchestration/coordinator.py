"""Single-writer leases, commits, pause, and crash recovery."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from engvit.orchestration.atomic import AtomicArtifactWriter
from engvit.orchestration.manifest import (
    ChunkRecord,
    CompletionRecord,
    JobState,
    Manifest,
    ManifestGeneration,
)
from engvit.types import ChunkCompletion, ChunkSpec


@dataclass(frozen=True)
class ChunkLease:
    chunk: ChunkSpec
    lease_id: str
    worker_id: str
    expires_at: float


@dataclass(frozen=True)
class RecoveryReport:
    generation: int
    repaired_chunks: tuple[str, ...]
    reset_chunks: tuple[str, ...]
    rejected_sidecars: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload(model: Manifest | CompletionRecord) -> bytes:
    return json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _completion_record(completion: ChunkCompletion) -> CompletionRecord:
    return CompletionRecord.model_validate(asdict(completion))


class Coordinator:
    """The only component authorized to advance a manifest generation."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        segments_root: Path,
        chunks: tuple[ChunkSpec, ...],
        job_identity_sha256: str,
        lease_seconds: int = 300,
        clock: Callable[[], float] = time.time,
        writer: AtomicArtifactWriter | None = None,
    ) -> None:
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        self._manifest_path = manifest_path
        self._segments_root = segments_root.resolve(strict=False)
        self._segments_root.mkdir(parents=True, exist_ok=True)
        self._specs = {chunk.chunk_id: chunk for chunk in chunks}
        if len(self._specs) != len(chunks):
            raise ValueError("chunk IDs must be unique")
        self._lease_seconds = lease_seconds
        self._clock = clock
        self._writer = writer or AtomicArtifactWriter()
        self._lock = threading.RLock()
        self._manifest_sha256: str | None = None

        if manifest_path.exists():
            loaded = Manifest.model_validate_json(manifest_path.read_bytes())
            if loaded.job_identity_sha256 != job_identity_sha256:
                raise ValueError("existing manifest belongs to another job identity")
            if {
                item.chunk_id: item.identity_sha256 for item in loaded.chunks
            } != {
                item.chunk_id: item.identity_sha256 for item in chunks
            }:
                raise ValueError("existing manifest chunk identities do not match")
            self._current = loaded
            self._manifest_sha256 = _sha256(manifest_path)
        else:
            self._current = Manifest(
                generation=0,
                job_identity_sha256=job_identity_sha256,
                state="running",
                chunks=tuple(
                    ChunkRecord(
                        chunk_id=chunk.chunk_id,
                        identity_sha256=chunk.identity_sha256,
                    )
                    for chunk in chunks
                ),
            )
            receipt = self._writer.write(manifest_path, _payload(self._current))
            self._manifest_sha256 = receipt.sha256

    @property
    def current(self) -> Manifest:
        with self._lock:
            return self._current

    @property
    def chunk_identities(self) -> dict[str, str]:
        return {
            chunk_id: spec.identity_sha256
            for chunk_id, spec in self._specs.items()
        }

    @property
    def segments_root(self) -> Path:
        return self._segments_root

    def _persist(
        self,
        chunks: tuple[ChunkRecord, ...],
        state: JobState,
    ) -> ManifestGeneration:
        updated = Manifest(
            generation=self._current.generation + 1,
            job_identity_sha256=self._current.job_identity_sha256,
            state=state,
            chunks=chunks,
        )
        receipt = self._writer.write(
            self._manifest_path,
            _payload(updated),
            expected_previous_sha256=self._manifest_sha256,
        )
        self._current = updated
        self._manifest_sha256 = receipt.sha256
        return ManifestGeneration(
            generation=updated.generation,
            state=updated.state,
            manifest_sha256=receipt.sha256,
        )

    def _expire(
        self,
        records: tuple[ChunkRecord, ...],
    ) -> tuple[tuple[ChunkRecord, ...], bool]:
        now = self._clock()
        changed = False
        result: list[ChunkRecord] = []
        for record in records:
            if (
                record.status == "leased"
                and record.lease_expires_at is not None
                and record.lease_expires_at <= now
            ):
                result.append(
                    ChunkRecord(
                        chunk_id=record.chunk_id,
                        identity_sha256=record.identity_sha256,
                    )
                )
                changed = True
            else:
                result.append(record)
        return tuple(result), changed

    def lease(self, worker_id: str) -> ChunkLease | None:
        if not worker_id:
            raise ValueError("worker_id must not be empty")
        with self._lock:
            records, expired = self._expire(self._current.chunks)
            if self._current.state in {"pause_requested", "paused", "complete", "failed"}:
                if expired:
                    self._persist(records, self._current.state)
                return None
            pending_index = next(
                (
                    index
                    for index, record in enumerate(records)
                    if record.status == "pending"
                ),
                None,
            )
            if pending_index is None:
                if expired:
                    self._persist(records, self._current.state)
                return None
            lease_id = uuid.uuid4().hex
            expires_at = self._clock() + self._lease_seconds
            record = records[pending_index]
            leased = ChunkRecord(
                chunk_id=record.chunk_id,
                identity_sha256=record.identity_sha256,
                status="leased",
                worker_id=worker_id,
                lease_id=lease_id,
                lease_expires_at=expires_at,
            )
            updated = list(records)
            updated[pending_index] = leased
            self._persist(tuple(updated), "running")
            return ChunkLease(
                chunk=self._specs[record.chunk_id],
                lease_id=lease_id,
                worker_id=worker_id,
                expires_at=expires_at,
            )

    def _validate_completion(
        self,
        completion: CompletionRecord,
        record: ChunkRecord,
        *,
        require_lease: bool,
    ) -> None:
        spec = self._specs[completion.chunk_id]
        if completion.identity_sha256 != spec.identity_sha256:
            raise ValueError("completion chunk identity does not match")
        if require_lease and (
            record.status != "leased" or record.lease_id != completion.lease_id
        ):
            raise ValueError("completion lease is stale or does not match")
        resolved = completion.partial_path.resolve(strict=True)
        try:
            resolved.relative_to(self._segments_root)
        except ValueError as exc:
            raise ValueError("completion path is outside the segments root") from exc
        if completion.partial_path.is_symlink() or not resolved.is_file():
            raise ValueError("completion segment must be a regular contained file")
        if resolved.stat().st_size != completion.bytes:
            raise ValueError("completion segment size does not match")
        if _sha256(resolved) != completion.sha256:
            raise ValueError("completion segment hash does not match")
        expected_frames = spec.output_core_end - spec.output_core_start
        if completion.frame_count != expected_frames:
            raise ValueError("completion frame count does not match chunk core")
        if (
            completion.first_pts != spec.output_core_start
            or completion.last_pts != spec.output_core_end - 1
        ):
            raise ValueError("completion PTS range does not match chunk core")

    def _sidecar_path(self, completion: CompletionRecord) -> Path:
        return completion.partial_path.with_name(
            f"{completion.partial_path.name}.completion.json"
        )

    def stage_completion(self, completion: ChunkCompletion) -> Path:
        """Durably stage worker evidence before advancing the manifest."""
        with self._lock:
            record = next(
                (
                    item
                    for item in self._current.chunks
                    if item.chunk_id == completion.chunk_id
                ),
                None,
            )
            if record is None:
                raise ValueError("completion references an unknown chunk")
            normalized = _completion_record(completion)
            self._validate_completion(normalized, record, require_lease=True)
            path = self._sidecar_path(normalized)
            self._writer.write(path, _payload(normalized))
            return path

    def commit(self, completion: ChunkCompletion) -> ManifestGeneration:
        with self._lock:
            normalized = _completion_record(completion)
            record_index = next(
                (
                    index
                    for index, item in enumerate(self._current.chunks)
                    if item.chunk_id == completion.chunk_id
                ),
                None,
            )
            if record_index is None:
                raise ValueError("completion references an unknown chunk")
            record = self._current.chunks[record_index]
            self._validate_completion(normalized, record, require_lease=True)
            self._writer.write(
                self._sidecar_path(normalized),
                _payload(normalized),
            )
            complete = ChunkRecord(
                chunk_id=record.chunk_id,
                identity_sha256=record.identity_sha256,
                status="complete",
                lease_id=normalized.lease_id,
                completion=normalized,
            )
            records = list(self._current.chunks)
            records[record_index] = complete
            if self._current.state in {"pause_requested", "paused"}:
                state: JobState = "paused"
            elif all(item.status == "complete" for item in records):
                state = "complete"
            else:
                state = "running"
            return self._persist(tuple(records), state)

    def request_pause(self) -> ManifestGeneration:
        with self._lock:
            if self._current.state != "running":
                raise ValueError("pause can only be requested while running")
            has_active = any(
                record.status == "leased" for record in self._current.chunks
            )
            state: JobState = "pause_requested" if has_active else "paused"
            return self._persist(self._current.chunks, state)

    def resume(self) -> ManifestGeneration:
        """Explicitly reopen a paused manifest for new leases."""
        with self._lock:
            if self._current.state != "paused":
                raise ValueError("resume can only be requested while paused")
            return self._persist(self._current.chunks, "running")

    def recover(self) -> RecoveryReport:
        with self._lock:
            loaded = Manifest.model_validate_json(self._manifest_path.read_bytes())
            self._current = loaded
            self._manifest_sha256 = _sha256(self._manifest_path)
            records, _ = self._expire(loaded.chunks)
            repaired: list[str] = []
            reset: list[str] = []
            rejected: list[str] = []

            mutable = list(records)
            for index, record in enumerate(mutable):
                if record.status != "complete" or record.completion is None:
                    continue
                try:
                    self._validate_completion(
                        record.completion,
                        record,
                        require_lease=False,
                    )
                except (OSError, ValueError):
                    mutable[index] = ChunkRecord(
                        chunk_id=record.chunk_id,
                        identity_sha256=record.identity_sha256,
                    )
                    reset.append(record.chunk_id)

            for path in sorted(self._segments_root.rglob("*.completion.json")):
                try:
                    candidate = CompletionRecord.model_validate_json(path.read_bytes())
                    index = next(
                        i
                        for i, record in enumerate(mutable)
                        if record.chunk_id == candidate.chunk_id
                    )
                    record = mutable[index]
                    if record.status == "complete":
                        continue
                    self._validate_completion(
                        candidate,
                        record,
                        require_lease=record.status == "leased",
                    )
                    mutable[index] = ChunkRecord(
                        chunk_id=record.chunk_id,
                        identity_sha256=record.identity_sha256,
                        status="complete",
                        lease_id=candidate.lease_id,
                        completion=candidate,
                    )
                    repaired.append(record.chunk_id)
                except (OSError, StopIteration, ValueError):
                    rejected.append(path.name)

            if loaded.state == "pause_requested":
                state: JobState = "paused"
            elif all(record.status == "complete" for record in mutable):
                state = "complete"
            elif loaded.state in {"paused", "failed"}:
                state = loaded.state
            else:
                state = "running"
            changed = tuple(mutable) != loaded.chunks or state != loaded.state
            if changed:
                generation = self._persist(tuple(mutable), state).generation
            else:
                generation = loaded.generation
            return RecoveryReport(
                generation=generation,
                repaired_chunks=tuple(repaired),
                reset_chunks=tuple(reset),
                rejected_sidecars=tuple(rejected),
            )
