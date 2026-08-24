from __future__ import annotations

import asyncio
import hashlib
import signal
from collections import deque
from pathlib import Path
from typing import Any, cast

import grpc
import pytest
from google.protobuf.any_pb2 import Any as AnyMessage
from google.protobuf.timestamp_pb2 import Timestamp
from google.rpc.error_details_pb2 import ErrorInfo
from google.rpc.status_pb2 import Status

from tandemn.chunkmanager.v1 import chunk_manager_pb2, chunk_manager_pb2_grpc
from tandemn_worker import batch_driver
from tandemn_worker.config import (
    BatchWorkerConfig,
    ChunkManagerConfig,
    MetricsConfig,
    load_batch_driver_config,
)
from tandemn_worker.metrics import create_metrics_app


class FakeRpcError(grpc.RpcError):
    def __init__(
        self,
        code: grpc.StatusCode,
        details: str = "test failure",
        trailing_metadata: tuple[tuple[str, bytes], ...] | None = None,
    ) -> None:
        super().__init__()
        self._code = code
        self._details = details
        self._trailing_metadata = trailing_metadata

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details

    def trailing_metadata(self) -> tuple[tuple[str, bytes], ...] | None:
        return self._trailing_metadata


class FakeWorkerStub:
    def __init__(
        self,
        *,
        claims: list[object] | None = None,
        renewals: list[object] | None = None,
        completions: list[object] | None = None,
    ) -> None:
        self.claims = deque(claims or [])
        self.renewals = deque(renewals or [])
        self.completions = deque(completions or [])
        self.claim_requests: list[chunk_manager_pb2.ClaimChunksRequest] = []
        self.renewal_requests: list[chunk_manager_pb2.RenewLeasesRequest] = []
        self.completion_requests: list[chunk_manager_pb2.CompleteChunkRequest] = []

    async def ClaimChunks(  # noqa: N802
        self,
        request: chunk_manager_pb2.ClaimChunksRequest,
        *,
        timeout: float,
    ) -> chunk_manager_pb2.ClaimChunksResponse:
        del timeout
        self.claim_requests.append(request)
        return cast(chunk_manager_pb2.ClaimChunksResponse, self._resolve(self.claims))

    async def RenewLeases(  # noqa: N802
        self,
        request: chunk_manager_pb2.RenewLeasesRequest,
        *,
        timeout: float,
    ) -> chunk_manager_pb2.RenewLeasesResponse:
        del timeout
        self.renewal_requests.append(request)
        return cast(chunk_manager_pb2.RenewLeasesResponse, self._resolve(self.renewals))

    async def CompleteChunk(  # noqa: N802
        self,
        request: chunk_manager_pb2.CompleteChunkRequest,
        *,
        timeout: float,
    ) -> chunk_manager_pb2.CompleteChunkResponse:
        del timeout
        self.completion_requests.append(request)
        return cast(chunk_manager_pb2.CompleteChunkResponse, self._resolve(self.completions))

    @staticmethod
    def _resolve(responses: deque[object]) -> object:
        response = responses.popleft()
        if isinstance(response, BaseException):
            raise response
        return response


def worker_stub(fake: FakeWorkerStub) -> chunk_manager_pb2_grpc.WorkerServiceStub:
    return cast(chunk_manager_pb2_grpc.WorkerServiceStub, cast(Any, fake))


def timestamp(seconds: int) -> Timestamp:
    return Timestamp(seconds=seconds)


def chain_identity() -> chunk_manager_pb2.ChainIdentity:
    return chunk_manager_pb2.ChainIdentity(
        job_id="01K2H7M9NWV2Q8JGRF3B5TC6DX",
        rank_id="01K2H7M9NWV2Q8JGRF3B5TC6DY",
        chain_id=2,
    )


def chunk_manager_config() -> ChunkManagerConfig:
    return ChunkManagerConfig(
        address="chunk-manager:50051",
        job_id="01K2H7M9NWV2Q8JGRF3B5TC6DX",
        rank_id="01K2H7M9NWV2Q8JGRF3B5TC6DY",
        chain_id=2,
        rpc_timeout_seconds=1.0,
        no_chunk_backoff_seconds=1.0,
    )


def metrics_config() -> MetricsConfig:
    return MetricsConfig(host="127.0.0.1", port=9000, path="/metrics")


def worker_config() -> BatchWorkerConfig:
    return BatchWorkerConfig(
        num_local_chunks=1,
        vllm_base_url="http://127.0.0.1:8000",
        vllm_ready_timeout_seconds=600.0,
        vllm_ready_interval_seconds=3.0,
        vllm_health_timeout_seconds=1.0,
        vllm_request_timeout_seconds=120.0,
        max_inflight_prompts=1,
    )


def config_environment(**overrides: str) -> dict[str, str]:
    environ = {
        "TD_CHUNK_MANAGER_ADDRESS": "chunk-manager:50051",
        "TD_JOB_ID": "job-1",
        "TD_RANK_ID": "rank-1",
        "TD_CHAIN_ID": "2",
    }
    environ.update(overrides)
    return environ


def lease_state(
    input_ref: str = "/tmp/input.jsonl",
    *,
    next_renewal_at: float = 1_000_000_000.0,
) -> batch_driver.LeaseState:
    return batch_driver.LeaseState(
        chunk_id=7,
        generation=3,
        input_ref=input_ref,
        next_renewal_at=next_renewal_at,
    )


def rich_rpc_error(code: grpc.StatusCode, reason: str) -> FakeRpcError:
    details = "chain is not active"
    error_info = ErrorInfo(reason=reason, domain="chunkmanager.tandemn.com")
    packed = AnyMessage()
    packed.Pack(error_info)
    status = Status(code=code.value[0], message=details, details=[packed])
    return FakeRpcError(
        code,
        details,
        (("grpc-status-details-bin", status.SerializeToString()),),
    )


def test_load_batch_driver_config_uses_defaults() -> None:
    config = load_batch_driver_config(
        {
            "TD_CHUNK_MANAGER_ADDRESS": " chunk-manager:50051 ",
            "TD_JOB_ID": " job-1 ",
            "TD_RANK_ID": " rank-1 ",
            "TD_CHAIN_ID": " 2 ",
        }
    )

    assert config.chunk_manager == ChunkManagerConfig(
        address="chunk-manager:50051",
        job_id="job-1",
        rank_id="rank-1",
        chain_id=2,
        rpc_timeout_seconds=5.0,
        no_chunk_backoff_seconds=1.0,
    )
    assert config.metrics == MetricsConfig(
        host="0.0.0.0",
        port=9000,
        path="/metrics",
    )
    assert config.worker == BatchWorkerConfig(
        num_local_chunks=5,
        vllm_base_url="http://127.0.0.1:8000",
        vllm_ready_timeout_seconds=600.0,
        vllm_ready_interval_seconds=3.0,
        vllm_health_timeout_seconds=1.0,
        vllm_request_timeout_seconds=120.0,
        max_inflight_prompts=100,
    )


def test_load_batch_driver_config_uses_overrides_and_vllm_port_fallback() -> None:
    environ = config_environment(
        TD_CHUNK_MANAGER_RPC_TIMEOUT_SECONDS="7.5",
        TD_NO_CHUNK_BACKOFF_SECONDS="2.5",
        TD_NUM_LOCAL_CHUNK="8",
        TD_VLLM_PORT="8123",
        TD_VLLM_READY_TIMEOUT_SECONDS="30",
        TD_VLLM_READY_INTERVAL_SECONDS="0.5",
        TD_VLLM_HEALTH_TIMEOUT_SECONDS="2",
        TD_VLLM_REQUEST_TIMEOUT_SECONDS="240",
        TD_MAX_INFLIGHT_PROMPTS="16",
        TD_METRICS_HOST="127.0.0.1",
        TD_METRICS_PORT="9100",
        TD_METRICS_PATH="/internal/metrics",
    )

    config = load_batch_driver_config(environ)

    assert config.chunk_manager.rpc_timeout_seconds == 7.5
    assert config.chunk_manager.no_chunk_backoff_seconds == 2.5
    assert config.worker == BatchWorkerConfig(
        num_local_chunks=8,
        vllm_base_url="http://127.0.0.1:8123",
        vllm_ready_timeout_seconds=30.0,
        vllm_ready_interval_seconds=0.5,
        vllm_health_timeout_seconds=2.0,
        vllm_request_timeout_seconds=240.0,
        max_inflight_prompts=16,
    )
    assert config.metrics == MetricsConfig(
        host="127.0.0.1",
        port=9100,
        path="/internal/metrics",
    )

    environ["TD_VLLM_BASE_URL"] = "http://vllm.example:9001/"
    config = load_batch_driver_config(environ)
    assert config.worker.vllm_base_url == "http://vllm.example:9001/"


def test_load_batch_driver_config_reports_all_missing_required_values() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Missing required environment variables: TD_CHUNK_MANAGER_ADDRESS, "
            "TD_JOB_ID, TD_RANK_ID, TD_CHAIN_ID"
        ),
    ):
        load_batch_driver_config({})


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("TD_CHAIN_ID", "invalid", "TD_CHAIN_ID must be an integer"),
        ("TD_CHUNK_MANAGER_RPC_TIMEOUT_SECONDS", "invalid", "must be a number"),
        ("TD_NO_CHUNK_BACKOFF_SECONDS", "invalid", "must be a number"),
        ("TD_NUM_LOCAL_CHUNK", "invalid", "must be an integer"),
        ("TD_MAX_INFLIGHT_PROMPTS", "invalid", "must be an integer"),
        ("TD_METRICS_PORT", "invalid", "must be an integer"),
        ("TD_VLLM_READY_TIMEOUT_SECONDS", "invalid", "must be a number"),
        ("TD_VLLM_READY_INTERVAL_SECONDS", "invalid", "must be a number"),
        ("TD_VLLM_HEALTH_TIMEOUT_SECONDS", "invalid", "must be a number"),
        ("TD_VLLM_REQUEST_TIMEOUT_SECONDS", "invalid", "must be a number"),
        ("TD_CHAIN_ID", "-1", "TD_CHAIN_ID must be non-negative"),
        ("TD_CHUNK_MANAGER_RPC_TIMEOUT_SECONDS", "0", "must be positive"),
        ("TD_NO_CHUNK_BACKOFF_SECONDS", "0", "must be positive"),
        ("TD_NUM_LOCAL_CHUNK", "0", "must be positive"),
        ("TD_MAX_INFLIGHT_PROMPTS", "0", "must be positive"),
    ],
)
def test_load_batch_driver_config_rejects_invalid_values(
    name: str,
    value: str,
    message: str,
) -> None:
    environ = config_environment()
    environ[name] = value

    with pytest.raises(ValueError, match=message):
        load_batch_driver_config(environ)


@pytest.mark.asyncio
async def test_claim_chunk_preserves_lease_and_uses_server_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(batch_driver.time, "monotonic", lambda: 100.0)
    response = chunk_manager_pb2.ClaimChunksResponse(
        job_state=chunk_manager_pb2.JOB_STATE_RUNNING,
        leases=[
            chunk_manager_pb2.ChunkLease(
                chunk_id=7,
                generation=3,
                input_ref="file:///data/input.jsonl",
                expires_at=timestamp(130),
            )
        ],
        database_time=timestamp(100),
    )
    fake = FakeWorkerStub(claims=[response])
    chain = chain_identity()

    result = await batch_driver.claim_chunk(worker_stub(fake), chain, 4.0)

    assert result.job_state == chunk_manager_pb2.JOB_STATE_RUNNING
    assert result.lease is not None
    assert result.lease.chunk_id == 7
    assert result.lease.generation == 3
    assert result.lease.input_ref == "file:///data/input.jsonl"
    assert result.lease.next_renewal_at == 115.0
    assert len(fake.claim_requests) == 1
    assert fake.claim_requests[0].chain == chain
    assert fake.claim_requests[0].max_chunks == 1


@pytest.mark.asyncio
async def test_claim_chunk_returns_no_lease_when_running_has_no_work() -> None:
    response = chunk_manager_pb2.ClaimChunksResponse(
        job_state=chunk_manager_pb2.JOB_STATE_RUNNING,
    )
    fake = FakeWorkerStub(claims=[response])

    result = await batch_driver.claim_chunk(worker_stub(fake), chain_identity(), 4.0)

    assert result.job_state == chunk_manager_pb2.JOB_STATE_RUNNING
    assert result.lease is None


@pytest.mark.asyncio
async def test_generated_client_works_with_async_grpc_transport() -> None:
    requests: list[chunk_manager_pb2.ClaimChunksRequest] = []

    class WorkerServicer(chunk_manager_pb2_grpc.WorkerServiceServicer):
        async def ClaimChunks(  # noqa: N802
            self,
            request: chunk_manager_pb2.ClaimChunksRequest,
            context: grpc.aio.ServicerContext,
        ) -> chunk_manager_pb2.ClaimChunksResponse:
            del context
            requests.append(request)
            return chunk_manager_pb2.ClaimChunksResponse(
                job_state=chunk_manager_pb2.JOB_STATE_SUCCEEDED,
            )

    server = grpc.aio.server()
    chunk_manager_pb2_grpc.add_WorkerServiceServicer_to_server(WorkerServicer(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = chunk_manager_pb2_grpc.WorkerServiceStub(channel)

    try:
        result = await batch_driver.claim_chunk(stub, chain_identity(), 1.0)
    finally:
        await channel.close()
        await server.stop(None)

    assert result.job_state == chunk_manager_pb2.JOB_STATE_SUCCEEDED
    assert result.lease is None
    assert len(requests) == 1
    assert requests[0].max_chunks == 1


@pytest.mark.asyncio
async def test_lease_lifecycle_marks_stale_renewal() -> None:
    lease = lease_state(next_renewal_at=0)
    response = chunk_manager_pb2.RenewLeasesResponse(
        stale=[lease.reference()],
        database_time=timestamp(100),
    )
    fake = FakeWorkerStub(renewals=[response])

    await batch_driver.lease_lifecycle(worker_stub(fake), chain_identity(), lease, 1.0)

    assert lease.stale_event.is_set()
    assert lease.completion_result.result() is False
    assert len(fake.renewal_requests) == 1


@pytest.mark.asyncio
async def test_lease_lifecycle_completes_exact_output_metadata() -> None:
    lease = lease_state()
    fake = FakeWorkerStub(
        completions=[
            chunk_manager_pb2.CompleteChunkResponse(
                job_state=chunk_manager_pb2.JOB_STATE_RUNNING,
            )
        ]
    )
    chain = chain_identity()
    lifecycle = asyncio.create_task(
        batch_driver.lease_lifecycle(worker_stub(fake), chain, lease, 1.0)
    )
    artifact = batch_driver.OutputArtifact(
        uri="file:///tmp/output.jsonl",
        checksum="sha256:abc",
        size_bytes=12,
    )

    assert await batch_driver.mark_chunk_completed(lease, artifact, chain)
    await lifecycle

    assert len(fake.completion_requests) == 1
    request = fake.completion_requests[0]
    assert request.chain == chain
    assert request.lease == lease.reference()
    assert request.output_uri == artifact.uri
    assert request.checksum == artifact.checksum
    assert request.output_size_bytes == artifact.size_bytes


@pytest.mark.asyncio
async def test_completion_renews_after_uncertain_result_and_replays_exact_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(batch_driver, "jittered_delay", lambda _delay: 0.0)
    lease = lease_state(next_renewal_at=0)
    renewal = chunk_manager_pb2.RenewLeasesResponse(
        renewed=[
            chunk_manager_pb2.RenewedLease(
                lease=lease.reference(),
                expires_at=timestamp(130),
            )
        ],
        database_time=timestamp(100),
    )
    fake = FakeWorkerStub(
        renewals=[renewal],
        completions=[
            FakeRpcError(grpc.StatusCode.UNAVAILABLE),
            chunk_manager_pb2.CompleteChunkResponse(
                job_state=chunk_manager_pb2.JOB_STATE_SUCCEEDED,
                replayed=True,
            ),
        ],
    )
    chain = chain_identity()
    request = chunk_manager_pb2.CompleteChunkRequest(
        chain=chain,
        lease=lease.reference(),
        output_uri="file:///tmp/output.jsonl",
        checksum="sha256:abc",
        output_size_bytes=12,
    )

    assert await batch_driver.complete_chunk(worker_stub(fake), chain, lease, request, 1.0)

    assert len(fake.renewal_requests) == 1
    assert len(fake.completion_requests) == 2
    assert fake.completion_requests[0] is request
    assert fake.completion_requests[1] is request


@pytest.mark.asyncio
async def test_slow_completion_is_interrupted_for_renewal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(batch_driver, "jittered_delay", lambda _delay: 0.0)
    lease = lease_state(next_renewal_at=batch_driver.time.monotonic() + 0.01)
    renewal = chunk_manager_pb2.RenewLeasesResponse(
        renewed=[
            chunk_manager_pb2.RenewedLease(
                lease=lease.reference(),
                expires_at=timestamp(130),
            )
        ],
        database_time=timestamp(100),
    )

    class SlowCompleteStub(FakeWorkerStub):
        async def CompleteChunk(  # noqa: N802
            self,
            request: chunk_manager_pb2.CompleteChunkRequest,
            *,
            timeout: float,
        ) -> chunk_manager_pb2.CompleteChunkResponse:
            del timeout
            self.completion_requests.append(request)
            if len(self.completion_requests) == 1:
                await asyncio.sleep(1)
            return chunk_manager_pb2.CompleteChunkResponse(
                job_state=chunk_manager_pb2.JOB_STATE_SUCCEEDED,
            )

    fake = SlowCompleteStub(renewals=[renewal])
    chain = chain_identity()
    request = chunk_manager_pb2.CompleteChunkRequest(
        chain=chain,
        lease=lease.reference(),
        output_uri="file:///tmp/output.jsonl",
        checksum="sha256:abc",
        output_size_bytes=12,
    )

    assert await batch_driver.complete_chunk(worker_stub(fake), chain, lease, request, 0.5)

    assert len(fake.completion_requests) == 2
    assert len(fake.renewal_requests) == 1


@pytest.mark.asyncio
async def test_not_found_renewal_is_a_stale_lease() -> None:
    fake = FakeWorkerStub(renewals=[FakeRpcError(grpc.StatusCode.NOT_FOUND)])

    renewed = await batch_driver.renew_lease(
        worker_stub(fake),
        chain_identity(),
        lease_state(),
        1.0,
    )

    assert not renewed


@pytest.mark.asyncio
async def test_stale_completion_does_not_retry_renewal() -> None:
    lease = lease_state()
    fake = FakeWorkerStub(
        completions=[rich_rpc_error(grpc.StatusCode.FAILED_PRECONDITION, "STALE_LEASE")]
    )
    chain = chain_identity()
    request = chunk_manager_pb2.CompleteChunkRequest(
        chain=chain,
        lease=lease.reference(),
        output_uri="file:///tmp/output.jsonl",
        checksum="sha256:abc",
        output_size_bytes=12,
    )

    completed = await batch_driver.complete_chunk(
        worker_stub(fake),
        chain,
        lease,
        request,
        1.0,
    )

    assert not completed
    assert not fake.renewal_requests


@pytest.mark.asyncio
async def test_chain_not_active_stops_claiming_and_drains() -> None:
    error = rich_rpc_error(grpc.StatusCode.FAILED_PRECONDITION, "CHAIN_NOT_ACTIVE")
    assert batch_driver.rpc_error_reason(error) == "CHAIN_NOT_ACTIVE"

    fake = FakeWorkerStub(claims=[error])
    chunk_sem = asyncio.BoundedSemaphore(1)
    input_queue: asyncio.Queue[batch_driver.InputChunk] = asyncio.Queue(maxsize=1)
    output_queue: asyncio.Queue[batch_driver.OutputChunk] = asyncio.Queue(maxsize=1)
    runtime = batch_driver.DriverRuntime(asyncio.Event(), asyncio.Event())
    intake_done = asyncio.Event()
    _, metrics = create_metrics_app(metrics_config())

    await batch_driver.chunk_puller(
        chunk_sem,
        input_queue,
        output_queue,
        runtime,
        intake_done,
        metrics,
        worker_stub(fake),
        chain_identity(),
        chunk_manager_config(),
        set(),
    )

    assert runtime.shutdown_event.is_set()
    assert runtime.is_draining
    assert not runtime.abort_event.is_set()
    assert runtime.return_code == int(batch_driver.DriverRuntime.ReturnCode.CHAIN_NOT_ACTIVE)
    assert intake_done.is_set()
    await asyncio.wait_for(chunk_sem.acquire(), timeout=0.1)


@pytest.mark.asyncio
async def test_draining_chain_completes_already_claimed_empty_chunk(tmp_path: Path) -> None:
    input_path = tmp_path / "empty.jsonl"
    input_path.write_text("", encoding="utf-8")
    claim = chunk_manager_pb2.ClaimChunksResponse(
        job_state=chunk_manager_pb2.JOB_STATE_RUNNING,
        leases=[
            chunk_manager_pb2.ChunkLease(
                chunk_id=7,
                generation=3,
                input_ref=input_path.as_uri(),
                expires_at=timestamp(130),
            )
        ],
        database_time=timestamp(100),
    )
    fake = FakeWorkerStub(
        claims=[
            claim,
            rich_rpc_error(grpc.StatusCode.FAILED_PRECONDITION, "CHAIN_NOT_ACTIVE"),
        ],
        completions=[
            chunk_manager_pb2.CompleteChunkResponse(
                job_state=chunk_manager_pb2.JOB_STATE_SUCCEEDED,
            )
        ],
    )
    chain = chain_identity()
    chunk_sem = asyncio.BoundedSemaphore(1)
    input_queue: asyncio.Queue[batch_driver.InputChunk] = asyncio.Queue(maxsize=1)
    output_queue: asyncio.Queue[batch_driver.OutputChunk] = asyncio.Queue(maxsize=1)
    runtime = batch_driver.DriverRuntime(asyncio.Event(), asyncio.Event())
    intake_done = asyncio.Event()
    lease_tasks: set[asyncio.Task[None]] = set()
    _, metrics = create_metrics_app(metrics_config())
    writer = asyncio.create_task(batch_driver.chunk_writer(output_queue, metrics, chain))

    try:
        await batch_driver.chunk_puller(
            chunk_sem,
            input_queue,
            output_queue,
            runtime,
            intake_done,
            metrics,
            worker_stub(fake),
            chain,
            chunk_manager_config(),
            lease_tasks,
        )
        await asyncio.wait_for(output_queue.join(), timeout=1)
        while lease_tasks:
            await asyncio.gather(*tuple(lease_tasks))
    finally:
        writer.cancel()
        await asyncio.gather(writer, return_exceptions=True)

    assert runtime.is_draining
    assert runtime.return_code == int(batch_driver.DriverRuntime.ReturnCode.CHAIN_NOT_ACTIVE)
    assert intake_done.is_set()
    assert len(fake.completion_requests) == 1
    assert fake.completion_requests[0].lease.chunk_id == 7
    assert fake.completion_requests[0].lease.generation == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "job_state",
    [
        chunk_manager_pb2.JOB_STATE_SUCCEEDED,
        chunk_manager_pb2.JOB_STATE_FAILED,
        chunk_manager_pb2.JOB_STATE_CANCELLED,
    ],
)
async def test_terminal_job_discards_local_work_without_drain(job_state: int) -> None:
    fake = FakeWorkerStub(
        claims=[
            chunk_manager_pb2.ClaimChunksResponse(
                job_state=job_state,
            )
        ]
    )
    chunk_sem = asyncio.BoundedSemaphore(1)
    runtime = batch_driver.DriverRuntime(asyncio.Event(), asyncio.Event())
    intake_done = asyncio.Event()
    _, metrics = create_metrics_app(metrics_config())

    await batch_driver.chunk_puller(
        chunk_sem,
        asyncio.Queue(maxsize=1),
        asyncio.Queue(maxsize=1),
        runtime,
        intake_done,
        metrics,
        worker_stub(fake),
        chain_identity(),
        chunk_manager_config(),
        set(),
    )

    assert runtime.shutdown_event.is_set()
    assert not runtime.is_draining
    assert runtime.abort_event.is_set()
    assert runtime.return_code == 0


@pytest.mark.asyncio
async def test_immediate_shutdown_escalates_an_existing_drain() -> None:
    runtime = batch_driver.DriverRuntime()

    runtime.handle_signal(signal.SIGTERM)
    runtime.handle_signal(signal.SIGUSR1)

    assert runtime.shutdown_event.is_set()
    assert runtime.abort_event.is_set()
    assert not runtime.is_draining
    assert runtime.return_code == 128 + signal.SIGUSR1


@pytest.mark.asyncio
async def test_shutdown_state_transitions_are_monotonic() -> None:
    runtime = batch_driver.DriverRuntime()

    assert not runtime.shutdown_event.is_set()
    assert not runtime.abort_event.is_set()
    assert not runtime.is_draining

    runtime.request_drain(20)

    assert runtime.shutdown_event.is_set()
    assert not runtime.abort_event.is_set()
    assert runtime.is_draining
    assert runtime.return_code == 20

    runtime.request_drain(21)
    assert runtime.return_code == 20

    runtime.request_abort(22)

    assert runtime.shutdown_event.is_set()
    assert runtime.abort_event.is_set()
    assert not runtime.is_draining
    assert runtime.return_code == 22

    runtime.request_drain(23)
    runtime.request_abort(24)
    assert runtime.return_code == 22


@pytest.mark.asyncio
@pytest.mark.parametrize("draining", [False, True])
async def test_supervised_task_failure_requests_abort(draining: bool) -> None:
    runtime = batch_driver.DriverRuntime()
    if draining:
        runtime.request_drain(20)

    async def fail() -> None:
        raise RuntimeError("test failure")

    await batch_driver.run_supervised("failing-worker", fail, runtime)

    assert runtime.shutdown_event.is_set()
    assert runtime.abort_event.is_set()
    assert runtime.return_code == int(batch_driver.DriverRuntime.ReturnCode.GENERIC_ERROR)


@pytest.mark.asyncio
async def test_supervised_task_rejects_unexpected_early_exit() -> None:
    runtime = batch_driver.DriverRuntime()

    async def finish() -> None:
        return

    await batch_driver.run_supervised("worker", finish, runtime)

    assert runtime.shutdown_event.is_set()
    assert runtime.abort_event.is_set()
    assert runtime.return_code == int(batch_driver.DriverRuntime.ReturnCode.GENERIC_ERROR)


@pytest.mark.asyncio
async def test_supervised_task_allows_expected_early_exit() -> None:
    runtime = batch_driver.DriverRuntime()

    async def finish() -> None:
        return

    await batch_driver.run_supervised(
        "lease",
        finish,
        runtime,
        allow_early_exit=True,
    )

    assert not runtime.shutdown_event.is_set()
    assert not runtime.abort_event.is_set()


@pytest.mark.asyncio
async def test_abort_interrupts_an_active_drain() -> None:
    runtime = batch_driver.DriverRuntime()
    drain_started = asyncio.Event()
    drain_cancelled = asyncio.Event()

    async def blocked_drain() -> None:
        drain_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            drain_cancelled.set()

    runtime.request_drain(0)
    task = asyncio.create_task(
        batch_driver.drain_until_aborted(blocked_drain(), runtime.abort_event)
    )
    await drain_started.wait()

    runtime.request_abort(1)
    await asyncio.wait_for(task, timeout=0.1)

    assert drain_cancelled.is_set()


@pytest.mark.asyncio
async def test_drain_waits_for_output_and_lease_tasks() -> None:
    async def finish() -> None:
        return

    lease_finished = asyncio.Event()

    async def finish_lease() -> None:
        await lease_finished.wait()

    chunk_puller = asyncio.create_task(finish())
    prompt_driver = asyncio.create_task(finish())
    lease_task = asyncio.create_task(finish_lease())
    output_queue: asyncio.Queue[batch_driver.OutputChunk] = asyncio.Queue()
    await output_queue.put(batch_driver.OutputChunk(lease=lease_state()))
    drain_task = asyncio.create_task(
        batch_driver.drain_worker(
            chunk_puller,
            prompt_driver,
            output_queue,
            {lease_task},
        )
    )

    try:
        await asyncio.sleep(0)
        assert not drain_task.done()

        output_queue.task_done()
        await asyncio.sleep(0)
        assert not drain_task.done()

        lease_finished.set()
        await asyncio.wait_for(drain_task, timeout=0.1)
    finally:
        lease_finished.set()
        for task in (chunk_puller, prompt_driver, lease_task, drain_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(
            chunk_puller,
            prompt_driver,
            lease_task,
            drain_task,
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_local_file_adapter_writes_immutable_generation_output(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("first\nsecond\n", encoding="utf-8")
    lease = lease_state(input_path.as_uri())

    input_chunk = await batch_driver.get_chunk(lease)
    assert input_chunk.prompts == ["first", "second"]

    output_chunk = batch_driver.OutputChunk(lease=lease, outputs=["one", "two"])
    artifact = await batch_driver.put_chunk(output_chunk, chain_identity().job_id)
    output_path = (
        tmp_path
        / "jobs"
        / chain_identity().job_id
        / "chunks"
        / "7"
        / "generations"
        / "3"
        / "output.jsonl"
    )
    expected = b"one\ntwo\n"

    assert output_path.read_bytes() == expected
    assert artifact.uri == output_path.as_uri()
    assert artifact.checksum == f"sha256:{hashlib.sha256(expected).hexdigest()}"
    assert artifact.size_bytes == len(expected)
    assert await batch_driver.put_chunk(output_chunk, chain_identity().job_id) == artifact

    concurrent_lease = batch_driver.LeaseState(
        chunk_id=7,
        generation=4,
        input_ref=input_path.as_uri(),
        next_renewal_at=1_000_000_000.0,
        input_path=input_path,
    )
    concurrent_chunk = batch_driver.OutputChunk(
        lease=concurrent_lease,
        outputs=["same"],
    )
    first, second = await asyncio.gather(
        batch_driver.put_chunk(concurrent_chunk, chain_identity().job_id),
        batch_driver.put_chunk(concurrent_chunk, chain_identity().job_id),
    )
    assert first == second

    with pytest.raises(RuntimeError, match="immutable output"):
        await batch_driver.put_chunk(
            batch_driver.OutputChunk(lease=lease, outputs=["different"]),
            chain_identity().job_id,
        )


def test_local_file_adapter_rejects_remote_schemes(tmp_path: Path) -> None:
    assert batch_driver.local_path_from_input_ref(str(tmp_path)) == tmp_path
    assert batch_driver.local_path_from_input_ref(tmp_path.as_uri()) == tmp_path

    with pytest.raises(NotImplementedError, match="S3 support is pending"):
        batch_driver.local_path_from_input_ref("s3://bucket/chunk.jsonl")
    with pytest.raises(ValueError, match="non-local authority"):
        batch_driver.local_path_from_input_ref("file://remote/chunk.jsonl")


@pytest.mark.asyncio
async def test_stale_queued_chunk_skips_vllm_and_releases_capacity(tmp_path: Path) -> None:
    lease = lease_state(str(tmp_path / "input.jsonl"))
    lease.stale_event.set()
    chunk = batch_driver.InputChunk(lease=lease, prompts=["unused"])
    chunk_sem = asyncio.BoundedSemaphore(1)
    await chunk_sem.acquire()
    input_queue: asyncio.Queue[batch_driver.InputChunk] = asyncio.Queue(maxsize=1)
    await input_queue.put(chunk)
    intake_done = asyncio.Event()
    intake_done.set()
    runtime = batch_driver.DriverRuntime(asyncio.Event(), asyncio.Event())
    _, metrics = create_metrics_app(metrics_config())

    await batch_driver.prompt_driver(
        chunk_sem,
        input_queue,
        asyncio.Queue(maxsize=1),
        intake_done,
        runtime,
        metrics,
        worker_config(),
    )

    await asyncio.wait_for(input_queue.join(), timeout=0.1)
    await asyncio.wait_for(chunk_sem.acquire(), timeout=0.1)
