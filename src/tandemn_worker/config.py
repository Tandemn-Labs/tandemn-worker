"""Batch driver configuration loaded from environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from tandemn.chunkmanager.v1 import chunk_manager_pb2


@dataclass(frozen=True, slots=True)
class ChunkManagerConfig:
    """Chunk manager connection, identity, and polling settings."""

    address: str
    job_id: str
    rank_id: str
    chain_id: int
    rpc_timeout_seconds: float
    no_chunk_backoff_seconds: float

    def chain_identity(self) -> chunk_manager_pb2.ChainIdentity:
        """Build the protobuf identity reused by all worker RPCs."""
        return chunk_manager_pb2.ChainIdentity(
            job_id=self.job_id,
            rank_id=self.rank_id,
            chain_id=self.chain_id,
        )


@dataclass(frozen=True, slots=True)
class MetricsConfig:
    """Metrics endpoint settings."""

    host: str
    port: int
    path: str


@dataclass(frozen=True, slots=True)
class BatchWorkerConfig:
    """Local chunk processing and vLLM client settings."""

    num_local_chunks: int
    vllm_base_url: str
    vllm_ready_timeout_seconds: float  # Max time to wait for vLLM engine to become ready
    vllm_ready_interval_seconds: float  # Delay between each check for vLLM readiness
    vllm_health_timeout_seconds: float  # HTTP timeout for readiness health check
    vllm_request_timeout_seconds: float  # HTTP timeout / request
    max_inflight_prompts: int


@dataclass(frozen=True, slots=True)
class BatchDriverConfig:
    """Complete configuration for one batch driver process."""

    chunk_manager: ChunkManagerConfig
    metrics: MetricsConfig
    worker: BatchWorkerConfig


def _read_int(environ: Mapping[str, str], name: str, default: str) -> int:
    value = environ.get(name, default)
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an int") from exc


def _read_float(environ: Mapping[str, str], name: str, default: str) -> float:
    value = environ.get(name, default)
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float") from exc


def load_batch_driver_config(
    environ: Mapping[str, str] | None = None,
) -> BatchDriverConfig:
    """Load and validate batch driver configuration from the environment."""
    if environ is None:
        environ = os.environ

    required_values = {
        "TD_CHUNK_MANAGER_ADDRESS": environ.get("TD_CHUNK_MANAGER_ADDRESS", "").strip(),
        "TD_JOB_ID": environ.get("TD_JOB_ID", "").strip(),
        "TD_RANK_ID": environ.get("TD_RANK_ID", "").strip(),
        "TD_CHAIN_ID": environ.get("TD_CHAIN_ID", "").strip(),
    }
    missing = [name for name, value in required_values.items() if not value]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    try:
        chain_id = int(required_values["TD_CHAIN_ID"])
    except ValueError as exc:
        raise ValueError("TD_CHAIN_ID must be an integer") from exc

    rpc_timeout_seconds = _read_float(
        environ,
        "TD_CHUNK_MANAGER_RPC_TIMEOUT_SECONDS",
        "10",
    )
    no_chunk_backoff_seconds = _read_float(
        environ,
        "TD_NO_CHUNK_BACKOFF_SECONDS",
        "10",
    )
    num_local_chunks = _read_int(environ, "TD_NUM_LOCAL_CHUNK", "5")
    max_inflight_prompts = _read_int(environ, "TD_MAX_INFLIGHT_PROMPTS", "100")

    if chain_id < 0:
        raise ValueError("TD_CHAIN_ID must be non-negative")
    if rpc_timeout_seconds <= 0:
        raise ValueError("TD_CHUNK_MANAGER_RPC_TIMEOUT_SECONDS must be positive")
    if no_chunk_backoff_seconds <= 0:
        raise ValueError("TD_NO_CHUNK_BACKOFF_SECONDS must be positive")
    if num_local_chunks <= 0:
        raise ValueError("TD_NUM_LOCAL_CHUNK must be positive")
    if max_inflight_prompts <= 0:
        raise ValueError("TD_MAX_INFLIGHT_PROMPTS must be positive")

    chunk_manager = ChunkManagerConfig(
        address=required_values["TD_CHUNK_MANAGER_ADDRESS"],
        job_id=required_values["TD_JOB_ID"],
        rank_id=required_values["TD_RANK_ID"],
        chain_id=chain_id,
        rpc_timeout_seconds=rpc_timeout_seconds,
        no_chunk_backoff_seconds=no_chunk_backoff_seconds,
    )
    metrics = MetricsConfig(
        host=environ.get("TD_METRICS_HOST", "0.0.0.0"),
        port=_read_int(environ, "TD_METRICS_PORT", "9000"),
        path=environ.get("TD_METRICS_PATH", "/metrics"),
    )
    worker = BatchWorkerConfig(
        num_local_chunks=num_local_chunks,
        vllm_base_url=environ.get(
            "TD_VLLM_BASE_URL",
            f"http://127.0.0.1:{environ.get('TD_VLLM_PORT', '8000')}",
        ),
        vllm_ready_timeout_seconds=_read_float(
            environ,
            "TD_VLLM_READY_TIMEOUT_SECONDS",
            "600",
        ),
        vllm_ready_interval_seconds=_read_float(
            environ,
            "TD_VLLM_READY_INTERVAL_SECONDS",
            "3",
        ),
        vllm_health_timeout_seconds=_read_float(
            environ,
            "TD_VLLM_HEALTH_TIMEOUT_SECONDS",
            "1",
        ),
        vllm_request_timeout_seconds=_read_float(
            environ,
            "TD_VLLM_REQUEST_TIMEOUT_SECONDS",
            "120",
        ),
        max_inflight_prompts=max_inflight_prompts,
    )
    return BatchDriverConfig(
        chunk_manager=chunk_manager,
        metrics=metrics,
        worker=worker,
    )
