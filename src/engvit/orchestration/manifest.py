"""Strict on-disk manifest schema."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

JobState = Literal[
    "running",
    "pause_requested",
    "paused",
    "complete",
    "failed",
]
ChunkState = Literal["pending", "leased", "complete"]


class CompletionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    lease_id: str
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    partial_path: Path
    bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_count: int = Field(ge=1)
    first_pts: int
    last_pts: int
    boundary_frame_hashes: tuple[str, ...]
    encoder_extradata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observations: dict[str, JsonValue]

    @field_validator("boundary_frame_hashes")
    @classmethod
    def validate_boundary_hashes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("completion requires boundary frame hashes")
        for value in values:
            if len(value) != 64:
                raise ValueError("boundary frame hashes must be SHA-256")
            try:
                bytes.fromhex(value)
            except ValueError as exc:
                raise ValueError("boundary frame hashes must be hexadecimal") from exc
        return values

    @model_validator(mode="after")
    def validate_bounds(self) -> CompletionRecord:
        if self.last_pts < self.first_pts:
            raise ValueError("completion PTS range is reversed")
        return self


class ChunkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: ChunkState = "pending"
    worker_id: str | None = None
    lease_id: str | None = None
    lease_expires_at: float | None = None
    completion: CompletionRecord | None = None

    @model_validator(mode="after")
    def state_fields_match(self) -> ChunkRecord:
        if self.status == "pending" and any(
            value is not None
            for value in (
                self.worker_id,
                self.lease_id,
                self.lease_expires_at,
                self.completion,
            )
        ):
            raise ValueError("pending chunk must not carry lease/completion fields")
        if self.status == "leased" and (
            self.worker_id is None
            or self.lease_id is None
            or self.lease_expires_at is None
            or self.completion is not None
        ):
            raise ValueError("leased chunk requires exactly its lease fields")
        if self.status == "complete" and (
            self.completion is None
            or self.worker_id is not None
            or self.lease_expires_at is not None
        ):
            raise ValueError("complete chunk requires only completion evidence")
        return self


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1"
    generation: int = Field(ge=0)
    job_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    state: JobState
    chunks: tuple[ChunkRecord, ...]


class ManifestGeneration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    generation: int
    state: JobState
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
