"""Batch inference driver entrypoint."""

import asyncio
import logging
import os
import signal
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, field
from enum import IntEnum
from typing import cast

import httpx

LOGGER = logging.getLogger(__name__)

# Chunk-related
NUM_LOCAL_CHUNK = int(os.getenv("TD_NUM_LOCAL_CHUNK", "5"))  # Limit for num chunks to hold locally
NO_CHUNK_BACKOFF_SECONDS = float(os.getenv("TD_NO_CHUNK_BACKOFF_SECONDS", "1"))

# vLLM-related
VLLM_BASE_URL = os.getenv(
    "TD_VLLM_BASE_URL",
    f"http://127.0.0.1:{os.getenv('TD_VLLM_PORT', '8000')}",
)
VLLM_READY_TIMEOUT_SECONDS = float(os.getenv("TD_VLLM_READY_TIMEOUT_SECONDS", "600"))
VLLM_READY_INTERVAL_SECONDS = float(os.getenv("TD_VLLM_READY_INTERVAL_SECONDS", "3"))
VLLM_HEALTH_TIMEOUT_SECONDS = float(os.getenv("TD_VLLM_HEALTH_TIMEOUT_SECONDS", "1"))
VLLM_REQUEST_TIMEOUT_SECONDS = float(os.getenv("TD_VLLM_REQUEST_TIMEOUT_SECONDS", "120"))
VLLM_MODEL = os.getenv("TD_VLLM_MODEL", "")
VLLM_MAX_TOKENS = int(os.getenv("TD_VLLM_MAX_TOKENS", "1024"))
MAX_INFLIGHT_PROMPTS = int(os.getenv("TD_MAX_INFLIGHT_PROMPTS", "100"))


@dataclass(slots=True)
class DriverRuntime:
    """Shared process-level shutdown state."""

    class ReturnCode(IntEnum):
        """Process return codes emitted by the batch driver."""

        SUCCESS = 0
        GENERIC_ERROR = 1
        VLLM_READY_TIMEOUT = 10

    shutdown_event: asyncio.Event
    return_code: int = int(ReturnCode.SUCCESS)
    should_drain_on_shutdown: bool = False

    def request_shutdown(
        self,
        signum: int | None = None,
        return_code: int | None = None,
    ) -> None:
        """Request driver shutdown and track whether locally claimed work drains."""
        signal_number = None if signum is None else int(signum)
        should_drain = signal_number in {
            int(signal.SIGINT),
            int(signal.SIGTERM),
        }

        if return_code is None:
            return_code = (
                128 + signal_number if signal_number is not None else int(self.ReturnCode.SUCCESS)
            )

        if should_drain and self.shutdown_event.is_set():
            return

        self.return_code = return_code
        self.should_drain_on_shutdown = should_drain
        self.shutdown_event.set()


@dataclass(slots=True)
class InputChunk:
    """A chunk claimed by this driver and ready for inference."""

    chunk_id: str
    prompts: list[str]


@dataclass(slots=True)
class OutputChunk:
    """A fully processed chunk ready to be written to external storage."""

    chunk_id: str
    outputs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ActiveChunk:
    """Prompt scheduling state for a chunk that has not been fully completed."""

    chunk_id: str
    prompts: list[str]
    outputs: list[str | None]
    next_prompt_index: int = 0
    remaining_prompts: int = 0

    @classmethod
    def from_input(cls, chunk: InputChunk) -> "ActiveChunk":
        """Create scheduling state for a newly pulled chunk."""
        if not chunk.prompts:
            raise ValueError(f"Active chunk {chunk.chunk_id} must have at least one prompt")

        return cls(
            chunk_id=chunk.chunk_id,
            prompts=chunk.prompts,
            outputs=[None] * len(chunk.prompts),
            remaining_prompts=len(chunk.prompts),
        )


@dataclass(slots=True)
class PromptResult:
    """A completed vLLM prompt response and its source chunk location."""

    chunk: ActiveChunk
    prompt_index: int
    output: str


async def main() -> int:
    """Run the batch driver event loop until shutdown is requested."""
    runtime = DriverRuntime(shutdown_event=asyncio.Event())
    loop = asyncio.get_running_loop()

    # SIGTERM / SIGINT - Attempt to do graceful drain
    # SIGUSR1 - Immediately kill all (usually when vLLM is already down)
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        loop.add_signal_handler(
            signum,
            runtime.request_shutdown,
            signum,
        )

    LOGGER.info("Batch driver started")

    chunk_sem = asyncio.BoundedSemaphore(NUM_LOCAL_CHUNK)
    input_chunk_queue: asyncio.Queue[InputChunk] = asyncio.Queue(maxsize=NUM_LOCAL_CHUNK)
    # Bound set to 3 times the input, could be changed.
    output_chunk_queue: asyncio.Queue[OutputChunk] = asyncio.Queue(maxsize=NUM_LOCAL_CHUNK * 3)

    chunk_puller_task = asyncio.create_task(
        chunk_puller(
            chunk_sem,
            input_chunk_queue,
            output_chunk_queue,
            runtime.shutdown_event,
        ),
        name="chunk-puller",
    )
    prompt_driver_task = asyncio.create_task(
        prompt_driver(chunk_sem, input_chunk_queue, output_chunk_queue, runtime),
        name="prompt-driver",
    )
    chunk_writer_task = asyncio.create_task(
        chunk_writer(output_chunk_queue),
        name="chunk-writer",
    )
    tasks = [chunk_puller_task, prompt_driver_task, chunk_writer_task]

    # Main blocking in main()
    await runtime.shutdown_event.wait()

    if runtime.should_drain_on_shutdown:
        LOGGER.info("Shutdown requested; draining locally claimed work")
        await chunk_puller_task
        await prompt_driver_task

        if runtime.should_drain_on_shutdown:
            await input_chunk_queue.join()
            await output_chunk_queue.join()
            chunk_writer_task.cancel()
            await asyncio.gather(chunk_writer_task, return_exceptions=True)
            LOGGER.info("Batch driver shutting down with return code %s", runtime.return_code)
            return runtime.return_code

    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)
    LOGGER.info("Batch driver shutting down with return code %s", runtime.return_code)
    return runtime.return_code


async def chunk_puller(
    chunk_sem: asyncio.BoundedSemaphore,
    input_chunk_queue: asyncio.Queue[InputChunk],
    output_chunk_queue: asyncio.Queue[OutputChunk],
    shutdown_event: asyncio.Event,
) -> None:
    """Claim chunks up to the local chunk limit without busy waiting."""
    while not shutdown_event.is_set():
        await chunk_sem.acquire()

        if shutdown_event.is_set():
            chunk_sem.release()
            break

        try:
            chunk = await get_chunk()
        except Exception:
            chunk_sem.release()
            raise

        if chunk is None:  # Chunk manager says no chunks, but job is ongoing
            chunk_sem.release()
            with suppress(TimeoutError):
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=NO_CHUNK_BACKOFF_SECONDS,
                )
            continue

        if not chunk.prompts:
            await output_chunk_queue.put(OutputChunk(chunk_id=chunk.chunk_id))
            chunk_sem.release()
            continue

        await input_chunk_queue.put(chunk)


async def prompt_driver(
    chunk_sem: asyncio.BoundedSemaphore,
    input_chunk_queue: asyncio.Queue[InputChunk],
    output_chunk_queue: asyncio.Queue[OutputChunk],
    runtime: DriverRuntime,
) -> None:
    """Submit chunk prompts to vLLM and enqueue completed chunks for writing."""
    shutdown_event = runtime.shutdown_event
    readiness_code = await wait_for_vllm_ready(runtime)
    if readiness_code != 0:
        # wait_for_vllm_ready returns 1 when vLLM did not become ready in time.
        if readiness_code == 1:
            LOGGER.error("Prompt driver stopping because vLLM readiness timed out")
            runtime.request_shutdown(return_code=int(DriverRuntime.ReturnCode.VLLM_READY_TIMEOUT))
        elif readiness_code == 2:
            # wait_for_vllm_ready returns 2 when shutdown was requested while waiting.
            LOGGER.info("Prompt driver stopping because shutdown was requested")
        return

    current_chunk: ActiveChunk | None = None
    inflight: set[asyncio.Task[PromptResult]] = set()

    async with httpx.AsyncClient(timeout=VLLM_REQUEST_TIMEOUT_SECONDS) as client:
        try:
            while True:
                while len(inflight) < MAX_INFLIGHT_PROMPTS:
                    # Load next chunk when current is finished
                    if current_chunk is None:
                        # Okay this section is convoluted but hold my beer
                        # Can afford to block getting an input chunk if there
                        # are no inflight request that might need processing
                        # However, it needs to be wakeable in case a shutdown signal
                        # comes. Therefore, you see what you see here.
                        if not inflight and not shutdown_event.is_set():
                            get_task = asyncio.create_task(input_chunk_queue.get())
                            shutdown_task = asyncio.create_task(shutdown_event.wait())
                            get_done, get_pending = await asyncio.wait(
                                {get_task, shutdown_task},
                                return_when=asyncio.FIRST_COMPLETED,
                            )

                            for pending_task in get_pending:
                                pending_task.cancel()
                            await asyncio.gather(*get_pending, return_exceptions=True)

                            if get_task in get_done:
                                input_chunk = get_task.result()
                            else:  # task that returned was the shutdown
                                break
                        # Some reqs are inflight OR shutdown_event.is_set()
                        else:
                            try:
                                input_chunk = input_chunk_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                        current_chunk = ActiveChunk.from_input(input_chunk)

                    # Get next prompt + chunk housekeeping
                    prompt_index = current_chunk.next_prompt_index
                    prompt = current_chunk.prompts[prompt_index]
                    current_chunk.next_prompt_index += 1

                    prompt_task = asyncio.create_task(
                        submit_prompt(client, current_chunk, prompt_index, prompt),
                        name=f"prompt-{current_chunk.chunk_id}-{prompt_index}",
                    )
                    inflight.add(prompt_task)

                    # So that current_chunk is not None when passed into task
                    if current_chunk.next_prompt_index >= len(current_chunk.prompts):
                        current_chunk = None

                # End of queueing more tasks, start of processing completed tasks

                if not inflight:
                    if (
                        shutdown_event.is_set()
                        and current_chunk is None
                        and input_chunk_queue.empty()
                    ):
                        return
                    continue

                done, pending = await asyncio.wait(
                    inflight,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                inflight = pending

                for task in done:
                    try:
                        result = task.result()
                    except Exception:
                        LOGGER.exception("Prompt request failed; stopping batch driver")
                        runtime.request_shutdown(
                            return_code=int(DriverRuntime.ReturnCode.GENERIC_ERROR)
                        )
                        return

                    chunk = result.chunk
                    chunk.outputs[result.prompt_index] = result.output
                    chunk.remaining_prompts -= 1

                    if chunk.remaining_prompts == 0:
                        await output_chunk_queue.put(
                            OutputChunk(
                                chunk_id=chunk.chunk_id,
                                outputs=cast(list[str], chunk.outputs),
                            )
                        )
                        input_chunk_queue.task_done()
                        chunk_sem.release()
        finally:
            for task in inflight:
                task.cancel()

            await asyncio.gather(*inflight, return_exceptions=True)


async def submit_prompt(
    client: httpx.AsyncClient,
    chunk: ActiveChunk,
    prompt_index: int,
    prompt: str,
) -> PromptResult:
    """Submit one prompt to vLLM and preserve its chunk location."""
    output = await call_vllm(client, prompt)
    return PromptResult(chunk=chunk, prompt_index=prompt_index, output=output)


async def call_vllm(client: httpx.AsyncClient, prompt: str) -> str:
    """Call the vLLM OpenAI-compatible completions endpoint."""
    response = await client.post(
        f"{VLLM_BASE_URL.rstrip('/')}/v1/completions",
        json={
            "model": VLLM_MODEL,
            "prompt": prompt,
            "max_tokens": VLLM_MAX_TOKENS,
        },
    )
    response.raise_for_status()

    response_body = response.json()
    if not isinstance(response_body, dict):
        raise RuntimeError("vLLM response body is not a JSON object")

    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("vLLM response body does not contain choices")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError("vLLM response choice is not a JSON object")

    text = choice.get("text")
    if not isinstance(text, str):
        raise RuntimeError("vLLM response choice does not contain text")

    return text


async def wait_for_vllm_ready(runtime: DriverRuntime) -> int:
    """Wait until the local vLLM OpenAI server is accepting requests."""
    shutdown_event = runtime.shutdown_event
    deadline = time.monotonic() + VLLM_READY_TIMEOUT_SECONDS
    health_url = f"{VLLM_BASE_URL.rstrip('/')}/health"

    LOGGER.info("Waiting for vLLM readiness at %s", health_url)

    async with httpx.AsyncClient(timeout=VLLM_HEALTH_TIMEOUT_SECONDS) as client:
        # Even if shutdown_event is set, keep waiting for vLLM if graceful drain.
        while not shutdown_event.is_set() or runtime.should_drain_on_shutdown:
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
            if runtime.should_drain_on_shutdown:
                await asyncio.sleep(VLLM_READY_INTERVAL_SECONDS)
            else:
                with suppress(TimeoutError):
                    await asyncio.wait_for(
                        shutdown_event.wait(),
                        timeout=VLLM_READY_INTERVAL_SECONDS,
                    )

    return 2


async def chunk_writer(output_chunk_queue: asyncio.Queue[OutputChunk]) -> None:
    """Write completed chunks to external storage."""
    while True:
        completed_chunk = await output_chunk_queue.get()

        try:
            await put_chunk(completed_chunk)
        finally:
            output_chunk_queue.task_done()  # For Queue.join()


async def get_chunk() -> InputChunk | None:
    """Claim the next chunk from the chunk manager and download it from storage."""
    # TODO
    return None


async def put_chunk(completed_chunk: OutputChunk) -> None:
    """Write a completed chunk to external storage."""
    # TODO
    return None


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

    raise SystemExit(asyncio.run(main()))
