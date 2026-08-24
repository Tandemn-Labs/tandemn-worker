"""Batch inference driver entrypoint."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import signal
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import suppress
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlparse

import grpc
import httpx
from google.protobuf.timestamp_pb2 import Timestamp
from google.rpc.error_details_pb2 import ErrorInfo
from grpc_status import rpc_status

from tandemn.chunkmanager.v1 import chunk_manager_pb2, chunk_manager_pb2_grpc
from tandemn_worker.config import (
    BatchDriverConfig,
    BatchWorkerConfig,
    ChunkManagerConfig,
    load_batch_driver_config,
)
from tandemn_worker.metrics import (
    DriverMetrics,
    create_metrics_app,
    start_metrics_server,
    stop_metrics_server,
)

LOGGER = logging.getLogger(__name__)

RPC_RETRY_INITIAL_SECONDS = 0.25
RPC_RETRY_MAX_SECONDS = 5.0
LEASE_RENEWAL_FRACTION = 0.5  # How much time used in lease before renewing
MIN_COMPLETION_RPC_SECONDS = 0.001
CHAIN_NOT_ACTIVE_REASON = "CHAIN_NOT_ACTIVE"
LEASE_EXPIRED_REASON = "LEASE_EXPIRED"
STALE_COMPLETION_REASONS = frozenset({"INVALID_STATE", "STALE_LEASE"})


########## Program lifecycle related classes and functions ##########


# Program is in 3 states: Running, Draining, Abort
# Running -> Draining -> Abort
# Running -> Abort
@dataclass(slots=True)
class DriverRuntime:
    """Shared process-level shutdown state."""

    class ReturnCode(IntEnum):
        """Process return codes emitted by the batch driver."""

        SUCCESS = 0
        GENERIC_ERROR = 1
        VLLM_READY_TIMEOUT = 10
        CHAIN_NOT_ACTIVE = 11

    shutdown_event: asyncio.Event = field(default_factory=asyncio.Event)
    abort_event: asyncio.Event = field(default_factory=asyncio.Event)
    return_code: int = int(ReturnCode.SUCCESS)

    @property
    def is_draining(self) -> bool:
        """Return whether shutdown is preserving already claimed work."""
        return self.shutdown_event.is_set() and not self.abort_event.is_set()

    def request_drain(self, return_code: int) -> None:
        """Begin shutdown while allowing already claimed work to complete."""

        # Continuous call to this function is no-op (Draining -> Draining)
        if self.shutdown_event.is_set():
            return

        self.return_code = return_code
        self.shutdown_event.set()

    def request_abort(self, return_code: int) -> None:
        """Begin or escalate shutdown without waiting for claimed work."""

        # Continuous call to this function is no-op (Abort -> Abort)
        if self.abort_event.is_set():
            return

        self.return_code = return_code
        self.abort_event.set()
        self.shutdown_event.set()

    def handle_signal(self, signum: int) -> None:
        """Translate a process signal into a drain or abort request."""
        signal_number = int(signum)
        return_code = 128 + signal_number

        # Drain only if SIGINT/SIGTERM, abort otherwise
        if signal_number in {int(signal.SIGINT), int(signal.SIGTERM)}:
            self.request_drain(return_code)
        else:
            self.request_abort(return_code)


# Shared infra to wrap all tasks, 2 main kinds of tasks are wrapped
# Core worker tasks, allow_early_exit = False
# lease lifecycle tasks, allow_early_exit = True
async def run_supervised(
    name: str,
    operation: Callable[[], Awaitable[None]],
    runtime: DriverRuntime,
    *,
    allow_early_exit: bool = False,  # Means "Can this task exit before proc shutdown"
) -> None:
    """Convert one task's failure or unexpected exit into a driver abort."""
    try:
        await operation()
    except asyncio.CancelledError:
        raise
    except Exception:
        LOGGER.exception("Background task %s failed", name)
        runtime.request_abort(int(DriverRuntime.ReturnCode.GENERIC_ERROR))
    else:  # Success case
        # allow_early_exit = True for lease lifecycle tasks.
        # allow_early_exit = False for core tasks, will trigger abort request
        if not allow_early_exit and not runtime.shutdown_event.is_set():
            LOGGER.error("Background task %s exited unexpectedly", name)
            runtime.request_abort(int(DriverRuntime.ReturnCode.GENERIC_ERROR))


async def drain_worker(
    chunk_puller_task: asyncio.Task[None],
    prompt_driver_task: asyncio.Task[None],
    output_chunk_queue: asyncio.Queue[OutputChunk],
    lease_tasks: set[asyncio.Task[None]],
) -> None:
    """Drain all locally claimed chunks while their leases remain active."""
    await chunk_puller_task
    await prompt_driver_task
    await output_chunk_queue.join()
    await asyncio.gather(*tuple(lease_tasks))


async def drain_until_aborted(
    drain: Coroutine[Any, Any, None],
    abort_event: asyncio.Event,
) -> None:
    """Wait for a graceful drain while allowing immediate escalation."""
    drain_task: asyncio.Task[None] = asyncio.create_task(drain, name="worker-drain")
    abort_task = asyncio.create_task(abort_event.wait(), name="abort-wait")
    helper_tasks = (drain_task, abort_task)
    try:
        done, _ = await asyncio.wait(
            helper_tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if abort_task in done:
            return
        await drain_task
    finally:  # All cases jump to here, which is a last guarantee
        for task in helper_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*helper_tasks, return_exceptions=True)


########## Core workers related ##########


@dataclass(slots=True)
class LeaseState:
    """Mutable local state for one claimed chunk generation."""

    chunk_id: int
    generation: int
    input_ref: str
    next_renewal_at: float
    stale_event: asyncio.Event = field(default_factory=asyncio.Event)
    completion_ready: asyncio.Event = field(default_factory=asyncio.Event)
    completion_request: chunk_manager_pb2.CompleteChunkRequest | None = None
    completion_result: asyncio.Future[bool] = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )
    input_path: Path | None = None
    lifecycle_task: asyncio.Task[None] | None = None

    def __post_init__(self) -> None:
        self.completion_result.add_done_callback(self._observe_completion_result)

    @staticmethod
    def _observe_completion_result(result: asyncio.Future[bool]) -> None:
        if not result.cancelled():
            result.exception()

    def reference(self) -> chunk_manager_pb2.LeaseReference:
        """Build an immutable reference to the currently owned generation."""
        return chunk_manager_pb2.LeaseReference(
            chunk_id=self.chunk_id,
            generation=self.generation,
        )


@dataclass(frozen=True, slots=True)
class ClaimResult:
    """Normalized result of one ClaimChunks RPC."""

    job_state: int
    lease: LeaseState | None = None


@dataclass(frozen=True, slots=True)
class OutputArtifact:
    """Metadata sent to the chunk manager after durable output publication."""

    uri: str
    checksum: str
    size_bytes: int


@dataclass(slots=True)
class InputChunk:
    """A chunk claimed by this driver and ready for inference."""

    lease: LeaseState
    prompts: list[str]

    @property
    def chunk_id(self) -> int:
        """Return the manager-assigned chunk ID."""
        return self.lease.chunk_id


@dataclass(slots=True)
class OutputChunk:
    """A fully processed chunk ready to be written to external storage."""

    lease: LeaseState
    outputs: list[str] = field(default_factory=list)

    @property
    def chunk_id(self) -> int:
        """Return the manager-assigned chunk ID."""
        return self.lease.chunk_id


@dataclass(slots=True)
class ActiveChunk:
    """Prompt scheduling state for a chunk that has not been fully completed."""

    lease: LeaseState
    prompts: list[str]
    outputs: list[str | None]
    next_prompt_index: int = 0
    remaining_prompts: int = 0
    finalized: bool = False

    @property
    def chunk_id(self) -> int:
        """Return the manager-assigned chunk ID."""
        return self.lease.chunk_id

    @classmethod
    def from_input(cls, chunk: InputChunk) -> ActiveChunk:
        """Create scheduling state for a newly pulled chunk."""
        if not chunk.prompts:
            raise ValueError(f"Active chunk {chunk.chunk_id} must have at least one prompt")

        return cls(
            lease=chunk.lease,
            prompts=chunk.prompts,
            outputs=[None] * len(chunk.prompts),
            remaining_prompts=len(chunk.prompts),
        )


@dataclass(slots=True)
class PromptResult:
    """A completed vLLM prompt response and its source chunk location."""

    chunk: ActiveChunk
    prompt_index: int
    output: str | None


TRANSIENT_RPC_CODES = frozenset(
    {
        grpc.StatusCode.ABORTED,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.UNAVAILABLE,
    }
)


def timestamp_delta_seconds(later: Timestamp, earlier: Timestamp) -> float:
    """Return the difference between two protobuf timestamps."""
    return (later.seconds - earlier.seconds) + (later.nanos - earlier.nanos) / 1_000_000_000


def next_renewal_at(
    call_started_at: float,
    expires_at: Timestamp,
    database_time: Timestamp,
) -> float:
    """Calculate a conservative monotonic renewal time from server timestamps."""
    lease_seconds = timestamp_delta_seconds(expires_at, database_time)
    if lease_seconds <= 0:
        return call_started_at
    return call_started_at + lease_seconds * LEASE_RENEWAL_FRACTION


def jittered_delay(delay_seconds: float) -> float:
    """Apply small jitter so many workers do not retry in lockstep."""
    return random.uniform(delay_seconds * 0.8, delay_seconds * 1.2)


def rpc_error_reason(error: grpc.RpcError) -> str | None:
    """Extract the chunk manager's stable ErrorInfo reason from an RPC failure."""
    status = rpc_status.from_call(cast(grpc.Call, error))
    if status is None:
        return None

    for detail in status.details:
        error_info = ErrorInfo()
        if detail.Unpack(error_info):
            return error_info.reason
    return None


async def wait_for_shutdown_or_timeout(event: asyncio.Event, delay_seconds: float) -> None:
    """Wait for shutdown while retaining an interruptible polling delay."""
    with suppress(TimeoutError):
        await asyncio.wait_for(event.wait(), timeout=jittered_delay(delay_seconds))


async def claim_chunk(
    stub: chunk_manager_pb2_grpc.WorkerServiceStub,
    chain: chunk_manager_pb2.ChainIdentity,
    rpc_timeout_seconds: float,
) -> ClaimResult:
    """Claim at most one chunk without transparently retrying the non-idempotent RPC."""
    call_started_at = time.monotonic()
    response = await stub.ClaimChunks(
        chunk_manager_pb2.ClaimChunksRequest(chain=chain, max_chunks=1),
        timeout=rpc_timeout_seconds,
    )

    if response.job_state != chunk_manager_pb2.JOB_STATE_RUNNING or not response.leases:
        return ClaimResult(job_state=int(response.job_state))

    claimed = response.leases[0]
    lease = LeaseState(
        chunk_id=claimed.chunk_id,
        generation=claimed.generation,
        input_ref=claimed.input_ref,
        next_renewal_at=next_renewal_at(
            call_started_at,
            claimed.expires_at,
            response.database_time,
        ),
    )
    return ClaimResult(job_state=int(response.job_state), lease=lease)


async def renew_lease(
    stub: chunk_manager_pb2_grpc.WorkerServiceStub,
    chain: chunk_manager_pb2.ChainIdentity,
    lease: LeaseState,
    rpc_timeout_seconds: float,
) -> bool:
    """Renew one lease until its authority is confirmed or definitively lost."""
    retry_delay = RPC_RETRY_INITIAL_SECONDS
    while True:
        call_started_at = time.monotonic()
        try:
            response = await stub.RenewLeases(
                chunk_manager_pb2.RenewLeasesRequest(
                    chain=chain,
                    leases=[lease.reference()],
                ),
                timeout=rpc_timeout_seconds,
            )
        except grpc.RpcError as exc:
            if exc.code() in {
                grpc.StatusCode.FAILED_PRECONDITION,
                grpc.StatusCode.NOT_FOUND,
            }:
                LOGGER.info(
                    "Lease can no longer be renewed for chunk=%s generation=%s: %s",
                    lease.chunk_id,
                    lease.generation,
                    exc.details(),
                )
                return False
            if exc.code() not in TRANSIENT_RPC_CODES:
                raise

            LOGGER.warning(
                "Lease renewal RPC failed for chunk=%s generation=%s; retrying: %s",
                lease.chunk_id,
                lease.generation,
                exc,
            )
            await asyncio.sleep(jittered_delay(retry_delay))
            retry_delay = min(retry_delay * 2, RPC_RETRY_MAX_SECONDS)
            continue

        stale = any(
            item.chunk_id == lease.chunk_id and item.generation == lease.generation
            for item in response.stale
        )
        renewed = next(
            (
                item
                for item in response.renewed
                if item.lease.chunk_id == lease.chunk_id
                and item.lease.generation == lease.generation
            ),
            None,
        )

        if stale and renewed is not None:
            raise RuntimeError("Chunk manager returned the same lease as renewed and stale")
        if stale:
            LOGGER.info(
                "Lease became stale for chunk=%s generation=%s",
                lease.chunk_id,
                lease.generation,
            )
            return False
        if renewed is None:
            raise RuntimeError("Chunk manager omitted the requested lease from renewal response")

        lease.next_renewal_at = next_renewal_at(
            call_started_at,
            renewed.expires_at,
            response.database_time,
        )
        return True


async def complete_chunk(
    stub: chunk_manager_pb2_grpc.WorkerServiceStub,
    chain: chunk_manager_pb2.ChainIdentity,
    lease: LeaseState,
    request: chunk_manager_pb2.CompleteChunkRequest,
    rpc_timeout_seconds: float,
) -> bool:
    """Complete a lease, replaying the exact request after uncertain outcomes."""
    retry_delay = RPC_RETRY_INITIAL_SECONDS
    outcome_uncertain = False
    renewal_rejected = False
    while True:
        if (
            not renewal_rejected
            and time.monotonic() + MIN_COMPLETION_RPC_SECONDS >= lease.next_renewal_at
        ):
            renewed = await renew_lease(stub, chain, lease, rpc_timeout_seconds)
            if not renewed:
                if not outcome_uncertain:
                    return False
                renewal_rejected = True

        call_timeout = rpc_timeout_seconds
        if not renewal_rejected:
            call_timeout = min(call_timeout, lease.next_renewal_at - time.monotonic())
            if call_timeout <= 0:
                continue

        try:
            response = await asyncio.wait_for(
                stub.CompleteChunk(request, timeout=call_timeout),
                timeout=call_timeout,
            )
        except TimeoutError:
            outcome_uncertain = True
            LOGGER.warning(
                "Completion RPC timed out for chunk=%s generation=%s; retrying",
                lease.chunk_id,
                lease.generation,
            )
        except grpc.RpcError as exc:
            if exc.code() in TRANSIENT_RPC_CODES:
                outcome_uncertain = True
                LOGGER.warning(
                    "Completion RPC failed for chunk=%s generation=%s; retrying: %s",
                    lease.chunk_id,
                    lease.generation,
                    exc,
                )
            elif exc.code() == grpc.StatusCode.NOT_FOUND:
                return False
            elif exc.code() == grpc.StatusCode.FAILED_PRECONDITION:
                if renewal_rejected:
                    return False
                reason = rpc_error_reason(exc)
                if reason in STALE_COMPLETION_REASONS:
                    return False
                if reason != LEASE_EXPIRED_REASON:
                    raise
                if await renew_lease(stub, chain, lease, rpc_timeout_seconds):
                    retry_delay = RPC_RETRY_INITIAL_SECONDS
                    continue
                return False
            else:
                raise
        else:
            LOGGER.info(
                "Completed chunk=%s generation=%s%s",
                lease.chunk_id,
                lease.generation,
                " by replay" if response.replayed else "",
            )
            return True

        sleep_seconds = jittered_delay(retry_delay)
        if not renewal_rejected:
            sleep_seconds = min(
                sleep_seconds,
                max(lease.next_renewal_at - time.monotonic(), 0.0),
            )
        await asyncio.sleep(sleep_seconds)
        retry_delay = min(
            retry_delay * 2,
            RPC_RETRY_MAX_SECONDS,
        )


async def lease_lifecycle(
    stub: chunk_manager_pb2_grpc.WorkerServiceStub,
    chain: chunk_manager_pb2.ChainIdentity,
    lease: LeaseState,
    rpc_timeout_seconds: float,
) -> None:
    """Renew a lease until output is ready, then serialize its completion."""
    try:
        while True:
            renewal_delay = max(lease.next_renewal_at - time.monotonic(), 0.0)
            try:
                await asyncio.wait_for(lease.completion_ready.wait(), timeout=renewal_delay)
            except TimeoutError:
                if await renew_lease(stub, chain, lease, rpc_timeout_seconds):
                    continue
                lease.stale_event.set()
                if not lease.completion_result.done():
                    lease.completion_result.set_result(False)
                return

            request = lease.completion_request
            if request is None:
                raise RuntimeError("Lease completion was requested without output metadata")

            completed = await complete_chunk(
                stub,
                chain,
                lease,
                request,
                rpc_timeout_seconds,
            )
            if not completed:
                lease.stale_event.set()
            if not lease.completion_result.done():
                lease.completion_result.set_result(completed)
            return
    except asyncio.CancelledError:
        if not lease.completion_result.done():
            lease.completion_result.cancel()
        raise
    except Exception as exc:
        if not lease.completion_result.done():
            lease.completion_result.set_exception(exc)
        raise


def start_lease_lifecycle(
    stub: chunk_manager_pb2_grpc.WorkerServiceStub,
    chain: chunk_manager_pb2.ChainIdentity,
    lease: LeaseState,
    rpc_timeout_seconds: float,
    lease_tasks: set[asyncio.Task[None]],
    runtime: DriverRuntime,
) -> None:
    """Start and supervise coordination for a newly claimed lease."""
    task_name = f"lease-{lease.chunk_id}-{lease.generation}"
    task = asyncio.create_task(
        run_supervised(
            task_name,
            lambda: lease_lifecycle(stub, chain, lease, rpc_timeout_seconds),
            runtime,
            allow_early_exit=True,
        ),
        name=task_name,
    )
    lease.lifecycle_task = task
    lease_tasks.add(task)
    task.add_done_callback(lease_tasks.discard)


async def cancel_lease_lifecycle(lease: LeaseState) -> None:
    """Stop renewing a claimed lease that cannot enter the local pipeline."""
    if lease.lifecycle_task is None:
        return
    lease.lifecycle_task.cancel()
    await asyncio.gather(lease.lifecycle_task, return_exceptions=True)


async def chunk_puller(
    chunk_sem: asyncio.BoundedSemaphore,
    input_chunk_queue: asyncio.Queue[InputChunk],
    output_chunk_queue: asyncio.Queue[OutputChunk],
    runtime: DriverRuntime,
    intake_done_event: asyncio.Event,
    metrics: DriverMetrics,
    stub: chunk_manager_pb2_grpc.WorkerServiceStub,
    chain: chunk_manager_pb2.ChainIdentity,
    config: ChunkManagerConfig,
    lease_tasks: set[asyncio.Task[None]],
) -> None:
    """Claim chunks up to the local chunk limit without busy waiting."""
    shutdown_event = runtime.shutdown_event
    try:
        while not shutdown_event.is_set():
            await chunk_sem.acquire()

            if shutdown_event.is_set():
                chunk_sem.release()
                break

            try:
                claim = await claim_chunk(stub, chain, config.rpc_timeout_seconds)
            except grpc.RpcError as exc:
                chunk_sem.release()
                if (
                    exc.code() == grpc.StatusCode.FAILED_PRECONDITION
                    and rpc_error_reason(exc) == CHAIN_NOT_ACTIVE_REASON
                ):
                    LOGGER.info("Chain is draining; stopping intake and completing local leases")
                    runtime.request_drain(int(DriverRuntime.ReturnCode.CHAIN_NOT_ACTIVE))
                    return
                if exc.code() not in TRANSIENT_RPC_CODES:
                    raise
                LOGGER.warning(
                    "Claim RPC failed with an uncertain outcome; any unseen lease will expire: %s",
                    exc,
                )
                await wait_for_shutdown_or_timeout(
                    shutdown_event,
                    config.no_chunk_backoff_seconds,
                )
                continue
            except Exception:
                chunk_sem.release()
                raise

            # Immediately abort since job is fully completed
            if claim.job_state != chunk_manager_pb2.JOB_STATE_RUNNING:
                chunk_sem.release()
                LOGGER.info("Chunk manager reported terminal job state %s", claim.job_state)
                runtime.request_abort(int(DriverRuntime.ReturnCode.SUCCESS))
                return

            lease = claim.lease
            # No lease but job is not complete, so wait and retry
            if lease is None:
                chunk_sem.release()
                await wait_for_shutdown_or_timeout(
                    shutdown_event,
                    config.no_chunk_backoff_seconds,
                )
                continue

            # Actually got a lease!
            start_lease_lifecycle(
                stub,
                chain,
                lease,
                config.rpc_timeout_seconds,
                lease_tasks,
                runtime,
            )
            try:
                chunk = await get_chunk(lease)
            except Exception:
                await cancel_lease_lifecycle(lease)
                chunk_sem.release()
                raise

            if lease.stale_event.is_set():
                chunk_sem.release()
                continue

            metrics.input_chunks_pulled.inc()

            if not chunk.prompts:
                await output_chunk_queue.put(OutputChunk(lease=lease))
                chunk_sem.release()
                continue

            await input_chunk_queue.put(chunk)
    finally:
        intake_done_event.set()


async def get_next_input_chunk(
    input_chunk_queue: asyncio.Queue[InputChunk],
    intake_done_event: asyncio.Event,
) -> InputChunk | None:
    """Wait for input until the puller confirms it can enqueue no more chunks."""
    if intake_done_event.is_set():
        try:
            return input_chunk_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    get_task = asyncio.create_task(input_chunk_queue.get())
    intake_done_task = asyncio.create_task(intake_done_event.wait())
    child_tasks = {get_task, intake_done_task}
    try:
        done, _ = await asyncio.wait(
            child_tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if get_task in done:
            return get_task.result()

        try:
            return input_chunk_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
    finally:
        for task in child_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*child_tasks, return_exceptions=True)


def discard_unscheduled_prompts(chunk: ActiveChunk) -> None:
    """Remove prompts that never started after a chunk loses its lease."""
    unscheduled = len(chunk.prompts) - chunk.next_prompt_index
    chunk.next_prompt_index = len(chunk.prompts)
    chunk.remaining_prompts -= unscheduled


def finalize_active_chunk(
    chunk: ActiveChunk,
    chunk_sem: asyncio.BoundedSemaphore,
    input_chunk_queue: asyncio.Queue[InputChunk],
) -> None:
    """Release input queue and semaphore ownership exactly once."""
    if chunk.finalized:
        return
    chunk.finalized = True
    input_chunk_queue.task_done()
    chunk_sem.release()


async def prompt_driver(
    chunk_sem: asyncio.BoundedSemaphore,
    input_chunk_queue: asyncio.Queue[InputChunk],
    output_chunk_queue: asyncio.Queue[OutputChunk],
    intake_done_event: asyncio.Event,
    runtime: DriverRuntime,
    metrics: DriverMetrics,
    config: BatchWorkerConfig,
) -> None:
    """Submit chunk prompts to vLLM and enqueue completed chunks for writing."""
    current_chunk: ActiveChunk | None = None
    inflight: set[asyncio.Task[PromptResult]] = set()
    vllm_ready = False

    async with httpx.AsyncClient(timeout=config.vllm_request_timeout_seconds) as client:
        try:
            while True:
                while len(inflight) < config.max_inflight_prompts:
                    if current_chunk is None:
                        # Okay this section is convoluted but hold my beer
                        # Can afford to block getting an input chunk if there
                        # are no inflight request that might need processing
                        # However, it needs to be wakeable in case a shutdown signal
                        # comes. Therefore, you see what you see here.
                        if not inflight:
                            input_chunk = await get_next_input_chunk(
                                input_chunk_queue,
                                intake_done_event,
                            )
                            if input_chunk is None:
                                return
                        else:
                            try:
                                input_chunk = input_chunk_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break

                        if input_chunk.lease.stale_event.is_set():
                            input_chunk_queue.task_done()
                            chunk_sem.release()
                            continue

                        if not vllm_ready:
                            readiness_code = await wait_for_vllm_ready(runtime, config)
                            if readiness_code != 0:
                                input_chunk_queue.task_done()
                                chunk_sem.release()
                                if readiness_code == 1:
                                    LOGGER.error(
                                        "Prompt driver stopping because vLLM readiness timed out"
                                    )
                                    runtime.request_abort(
                                        int(DriverRuntime.ReturnCode.VLLM_READY_TIMEOUT)
                                    )
                                else:
                                    LOGGER.info(
                                        "Prompt driver stopping because shutdown was requested"
                                    )
                                return
                            vllm_ready = True

                        if input_chunk.lease.stale_event.is_set():
                            input_chunk_queue.task_done()
                            chunk_sem.release()
                            continue
                        current_chunk = ActiveChunk.from_input(input_chunk)

                    if current_chunk.lease.stale_event.is_set():
                        discard_unscheduled_prompts(current_chunk)
                        if current_chunk.remaining_prompts == 0:
                            finalize_active_chunk(
                                current_chunk,
                                chunk_sem,
                                input_chunk_queue,
                            )
                        current_chunk = None
                        continue

                    prompt_index = current_chunk.next_prompt_index
                    prompt = current_chunk.prompts[prompt_index]
                    current_chunk.next_prompt_index += 1

                    prompt_task = asyncio.create_task(
                        submit_prompt(client, current_chunk, prompt_index, prompt, config),
                        name=(
                            f"prompt-{current_chunk.chunk_id}-"
                            f"{current_chunk.lease.generation}-{prompt_index}"
                        ),
                    )
                    inflight.add(prompt_task)
                    metrics.inflight_requests.set(len(inflight))

                    if current_chunk.next_prompt_index >= len(current_chunk.prompts):
                        current_chunk = None

                if not inflight:
                    if (
                        intake_done_event.is_set()
                        and current_chunk is None
                        and input_chunk_queue.empty()
                    ):
                        return
                    continue

                done, _ = await asyncio.wait(
                    inflight,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                metrics.inflight_requests.set(len(inflight) - len(done))

                for task in done:
                    result = task.result()
                    inflight.remove(task)

                    chunk = result.chunk
                    if chunk.finalized:
                        continue

                    chunk.remaining_prompts -= 1
                    if result.output is not None and not chunk.lease.stale_event.is_set():
                        metrics.requests_processed.inc()
                        chunk.outputs[result.prompt_index] = result.output

                    if chunk.lease.stale_event.is_set():
                        discard_unscheduled_prompts(chunk)
                        if current_chunk is chunk:
                            current_chunk = None

                    if chunk.remaining_prompts == 0:
                        if not chunk.lease.stale_event.is_set():
                            await output_chunk_queue.put(
                                OutputChunk(
                                    lease=chunk.lease,
                                    outputs=cast(list[str], chunk.outputs),
                                )
                            )
                        finalize_active_chunk(chunk, chunk_sem, input_chunk_queue)
        finally:
            for task in inflight:
                task.cancel()

            await asyncio.gather(*inflight, return_exceptions=True)
            metrics.inflight_requests.set(0)


async def submit_prompt(
    client: httpx.AsyncClient,
    chunk: ActiveChunk,
    prompt_index: int,
    prompt: str,
    config: BatchWorkerConfig,
) -> PromptResult:
    """Cancel a vLLM request promptly if its lease becomes stale."""
    if chunk.lease.stale_event.is_set():
        return PromptResult(chunk=chunk, prompt_index=prompt_index, output=None)

    request_task = asyncio.create_task(
        submit_prompt_request(client, chunk, prompt_index, prompt, config)
    )

    # Similar pattern of process prompt but break if stale_event is set,
    # which means the lease expired and processing the curr chunk is useless
    stale_task = asyncio.create_task(chunk.lease.stale_event.wait())
    child_tasks = {request_task, stale_task}
    try:
        done, _ = await asyncio.wait(
            child_tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if stale_task in done and chunk.lease.stale_event.is_set():
            return PromptResult(chunk=chunk, prompt_index=prompt_index, output=None)
        return request_task.result()
    finally:
        for task in child_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*child_tasks, return_exceptions=True)


async def submit_prompt_request(
    client: httpx.AsyncClient,
    chunk: ActiveChunk,
    prompt_index: int,
    prompt: str,
    config: BatchWorkerConfig,
) -> PromptResult:
    """Submit one Batch JSONL request to vLLM and preserve its chunk location."""
    request = json.loads(prompt)
    if not isinstance(request, dict):
        raise RuntimeError("Batch request line is not a JSON object")

    custom_id = request.get("custom_id")
    if not isinstance(custom_id, str):
        raise RuntimeError("Batch request does not contain custom_id")

    method = request.get("method")
    if method != "POST":
        raise RuntimeError("Batch request method must be POST")

    url = request.get("url")
    if not isinstance(url, str) or not url.startswith("/v1/"):
        raise RuntimeError("Batch request url must be a relative /v1/ path")

    body = request.get("body")
    if not isinstance(body, dict):
        raise RuntimeError("Batch request body is not a JSON object")

    response_payload: object | None
    error_payload: object | None

    try:
        response = await client.post(f"{config.vllm_base_url.rstrip('/')}{url}", json=body)
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        message = (
            str(exc) or f"Request timed out after {config.vllm_request_timeout_seconds:g} seconds"
        )
        response_payload = None
        error_payload = {
            "type": exc.__class__.__name__,
            "message": message,
        }
    except httpx.HTTPError as exc:
        response_payload = None
        error_payload = {
            "type": exc.__class__.__name__,
            "message": str(exc),
        }
    else:
        response_payload = {
            "status_code": response.status_code,
            "request_id": response.headers.get("x-request-id"),
            "body": response.json(),
        }
        error_payload = None

    output = {
        "id": f"batch_req_{chunk.chunk_id}_{chunk.lease.generation}_{prompt_index}",
        "custom_id": custom_id,
        "response": response_payload,
        "error": error_payload,
    }
    return PromptResult(
        chunk=chunk,
        prompt_index=prompt_index,
        output=json.dumps(output, separators=(",", ":")),
    )


async def wait_for_vllm_ready(runtime: DriverRuntime, config: BatchWorkerConfig) -> int:
    """Wait until the local vLLM OpenAI server is accepting requests."""
    shutdown_event = runtime.shutdown_event
    deadline = time.monotonic() + config.vllm_ready_timeout_seconds
    health_url = f"{config.vllm_base_url.rstrip('/')}/health"

    LOGGER.info("Waiting for vLLM readiness at %s", health_url)

    async with httpx.AsyncClient(timeout=config.vllm_health_timeout_seconds) as client:
        # Even if shutdown_event is set, keep waiting for vLLM if graceful drain.
        while not shutdown_event.is_set() or runtime.is_draining:
            if time.monotonic() >= deadline:
                return 1

            try:
                response = await client.get(health_url)
            except httpx.HTTPError as exc:
                LOGGER.info("vLLM is not ready yet: %s", exc)
            else:
                if response.status_code == 200:
                    LOGGER.info("vLLM is ready")
                    return 0
                LOGGER.info(
                    "vLLM is not ready yet: health returned HTTP %s",
                    response.status_code,
                )

            # If graceful drain, can't wait on shutdown_event anymore
            if runtime.is_draining:
                await asyncio.sleep(config.vllm_ready_interval_seconds)
            else:
                with suppress(TimeoutError):
                    await asyncio.wait_for(
                        shutdown_event.wait(),
                        timeout=config.vllm_ready_interval_seconds,
                    )

    return 2


async def chunk_writer(
    output_chunk_queue: asyncio.Queue[OutputChunk],
    metrics: DriverMetrics,
    chain: chunk_manager_pb2.ChainIdentity,
) -> None:
    """Publish outputs and wait for authoritative manager completion."""
    while True:
        completed_chunk = await output_chunk_queue.get()

        try:
            lease = completed_chunk.lease

            # Skip this chunk completion if the lease is invalid
            if lease.stale_event.is_set():
                continue

            artifact = await put_chunk(completed_chunk, chain.job_id)
            if lease.stale_event.is_set():
                continue

            completed = await mark_chunk_completed(lease, artifact, chain)
            if completed:
                metrics.output_chunks_written.inc()
        finally:
            output_chunk_queue.task_done()  # For Queue.join()


async def mark_chunk_completed(
    lease: LeaseState,
    artifact: OutputArtifact,
    chain: chunk_manager_pb2.ChainIdentity,
) -> bool:
    """Hand immutable output metadata to the lease lifecycle and await its result."""
    if lease.completion_request is not None:
        raise RuntimeError(
            f"Completion already requested for chunk={lease.chunk_id} generation={lease.generation}"
        )

    lease.completion_request = chunk_manager_pb2.CompleteChunkRequest(
        chain=chain,
        lease=lease.reference(),
        output_uri=artifact.uri,
        checksum=artifact.checksum,
        output_size_bytes=artifact.size_bytes,
    )
    lease.completion_ready.set()
    return await asyncio.shield(lease.completion_result)


def local_path_from_input_ref(input_ref: str) -> Path:
    """Resolve a temporary local path adapter for a manager input reference."""
    parsed = urlparse(input_ref)
    if not parsed.scheme:
        return Path(input_ref).expanduser().resolve()
    if parsed.scheme != "file":
        raise NotImplementedError(
            f"Input reference scheme {parsed.scheme!r} is not supported; S3 support is pending"
        )
    if parsed.netloc not in {"", "localhost"}:
        raise ValueError(f"File input reference has non-local authority {parsed.netloc!r}")
    return Path(unquote(parsed.path)).expanduser().resolve()


async def get_chunk(lease: LeaseState) -> InputChunk:
    """Read a claimed chunk through the temporary local-file storage adapter."""
    chunk_path = local_path_from_input_ref(lease.input_ref)
    lease.input_path = chunk_path
    prompts = await asyncio.to_thread(lambda: chunk_path.read_text(encoding="utf-8").splitlines())
    return InputChunk(lease=lease, prompts=prompts)


def fsync_directory(directory: Path) -> None:
    """Persist directory entry changes on the local filesystem."""
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_file_and_directory(path: Path) -> None:
    """Persist an existing output and the directory entry that names it."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)


def create_output_directories(root: Path, output_directory: Path) -> None:
    """Create and durably link each generation-specific output directory."""
    current = root
    for part in output_directory.relative_to(root).parts:
        child = current / part
        try:
            child.mkdir()
        except FileExistsError:
            if not child.is_dir():
                raise
        fsync_directory(current)
        current = child


def write_output_immutably(root: Path, output_path: Path, output_bytes: bytes) -> None:
    """Durably publish output without replacing an existing generation artifact."""
    create_output_directories(root, output_path.parent)
    if output_path.exists():
        if output_path.read_bytes() == output_bytes:
            fsync_file_and_directory(output_path)
            return
        raise RuntimeError(f"Refusing to overwrite immutable output {output_path}")

    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output_file:
            output_file.write(output_bytes)
            output_file.flush()
            os.fsync(output_file.fileno())

        try:
            os.link(temporary_path, output_path)
        except FileExistsError:
            if output_path.read_bytes() != output_bytes:
                raise RuntimeError(
                    f"Refusing to overwrite immutable output {output_path}"
                ) from None
        fsync_file_and_directory(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


async def put_chunk(completed_chunk: OutputChunk, job_id: str) -> OutputArtifact:
    """Write a completed chunk and return exact metadata for CompleteChunk."""
    lease = completed_chunk.lease
    if lease.input_path is None:
        raise RuntimeError(f"Chunk {lease.chunk_id} has no resolved input path")

    output_path = (
        lease.input_path.parent
        / "jobs"
        / job_id
        / "chunks"
        / str(lease.chunk_id)
        / "generations"
        / str(lease.generation)
        / "output.jsonl"
    )
    output_text = "\n".join(completed_chunk.outputs)
    if output_text:
        output_text += "\n"
    output_bytes = output_text.encode("utf-8")

    await asyncio.to_thread(
        write_output_immutably,
        lease.input_path.parent,
        output_path,
        output_bytes,
    )
    return OutputArtifact(
        uri=output_path.as_uri(),
        checksum=f"sha256:{hashlib.sha256(output_bytes).hexdigest()}",
        size_bytes=len(output_bytes),
    )


async def main(config: BatchDriverConfig) -> int:
    """Run the batch driver event loop until shutdown is requested."""
    runtime = DriverRuntime()
    loop = asyncio.get_running_loop()
    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1)

    ##### Housekeeping / Setup #####

    # SIGTERM / SIGINT attempt to drain; SIGUSR1 aborts because vLLM is already down.
    for signum in signals:
        loop.add_signal_handler(signum, runtime.handle_signal, signum)

    LOGGER.info(
        "Batch driver started for job=%s rank=%s chain=%s",
        config.chunk_manager.job_id,
        config.chunk_manager.rank_id,
        config.chunk_manager.chain_id,
    )

    metrics_app, metrics = create_metrics_app(config.metrics)
    metrics_server, metrics_server_task = start_metrics_server(metrics_app, config.metrics)

    ##### Variables for main workers #####

    chunk_sem = asyncio.BoundedSemaphore(config.worker.num_local_chunks)
    input_chunk_queue: asyncio.Queue[InputChunk] = asyncio.Queue(
        maxsize=config.worker.num_local_chunks
    )
    output_chunk_queue: asyncio.Queue[OutputChunk] = asyncio.Queue(
        maxsize=config.worker.num_local_chunks * 3
    )
    intake_done_event = asyncio.Event()
    lease_tasks: set[asyncio.Task[None]] = set()
    channel = grpc.aio.insecure_channel(
        config.chunk_manager.address,
        options=(("grpc.enable_retries", 0),),
    )
    stub = chunk_manager_pb2_grpc.WorkerServiceStub(channel)
    chain = config.chunk_manager.chain_identity()

    ##### Spawn main worker tasks #####

    chunk_puller_task = asyncio.create_task(
        run_supervised(
            "chunk-puller",
            lambda: chunk_puller(
                chunk_sem,
                input_chunk_queue,
                output_chunk_queue,
                runtime,
                intake_done_event,
                metrics,
                stub,
                chain,
                config.chunk_manager,
                lease_tasks,
            ),
            runtime,
        ),
        name="chunk-puller",
    )
    prompt_driver_task = asyncio.create_task(
        run_supervised(
            "prompt-driver",
            lambda: prompt_driver(
                chunk_sem,
                input_chunk_queue,
                output_chunk_queue,
                intake_done_event,
                runtime,
                metrics,
                config.worker,
            ),
            runtime,
        ),
        name="prompt-driver",
    )
    chunk_writer_task = asyncio.create_task(
        run_supervised(
            "chunk-writer",
            lambda: chunk_writer(output_chunk_queue, metrics, chain),
            runtime,
        ),
        name="chunk-writer",
    )
    tasks = [chunk_puller_task, prompt_driver_task, chunk_writer_task]

    try:
        await runtime.shutdown_event.wait()

        if runtime.is_draining:
            LOGGER.info("Shutdown requested; draining locally claimed work")
            try:
                await drain_until_aborted(
                    drain_worker(
                        chunk_puller_task,
                        prompt_driver_task,
                        output_chunk_queue,
                        lease_tasks,
                    ),
                    runtime.abort_event,
                )
            except Exception:
                LOGGER.exception("Failed while draining locally claimed work")
                runtime.request_abort(int(DriverRuntime.ReturnCode.GENERIC_ERROR))
    finally:
        tasks_to_stop = (*tasks, *tuple(lease_tasks))
        for task in tasks_to_stop:
            if not task.done():
                task.cancel()

        await asyncio.gather(*tasks_to_stop, return_exceptions=True)
        await channel.close()
        await stop_metrics_server(metrics_server, metrics_server_task)
        for signum in signals:
            loop.remove_signal_handler(signum)

    LOGGER.info("Batch driver shutting down with return code %s", runtime.return_code)
    return runtime.return_code


if __name__ == "__main__":
    # Initialize logger
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        force=True,
    )

    try:
        driver_config = load_batch_driver_config()
    except ValueError as exc:
        LOGGER.error("Invalid batch driver configuration: %s", exc)
        raise SystemExit(int(DriverRuntime.ReturnCode.GENERIC_ERROR)) from None

    raise SystemExit(asyncio.run(main(driver_config)))
