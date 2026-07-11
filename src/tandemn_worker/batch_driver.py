"""Batch inference driver entrypoint."""

import asyncio
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field

LOGGER = logging.getLogger(__name__)

# Chunk-related
NUM_LOCAL_CHUNK = int(os.getenv("TD_NUM_LOCAL_CHUNK", "5"))  # Limit for num chunks to hold locally
NO_CHUNK_BACKOFF_SECONDS = float(os.getenv("TD_NO_CHUNK_BACKOFF_SECONDS", "1"))


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


async def main() -> None:
    """Run the batch driver event loop until shutdown is requested."""
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_shutdown() -> None:
        # TODO: stop intake and flush safely completed work before exiting.
        shutdown_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, request_shutdown)

    LOGGER.info("Batch driver started")

    chunk_sem = asyncio.BoundedSemaphore(NUM_LOCAL_CHUNK)
    input_chunk_queue: asyncio.Queue[InputChunk] = asyncio.Queue(maxsize=NUM_LOCAL_CHUNK)
    # Bound set to 3 times the input, could be changed.
    output_chunk_queue: asyncio.Queue[OutputChunk] = asyncio.Queue(maxsize=NUM_LOCAL_CHUNK * 3)

    tasks = [
        asyncio.create_task(
            chunk_puller(chunk_sem, input_chunk_queue, shutdown_event),
            name="chunk-puller",
        ),
        asyncio.create_task(
            prompt_driver(chunk_sem, input_chunk_queue, output_chunk_queue, shutdown_event),
            name="prompt-driver",
        ),
        asyncio.create_task(
            chunk_writer(output_chunk_queue, shutdown_event),
            name="chunk-writer",
        ),
    ]

    await (
        shutdown_event.wait()
    )  # ERROR: shouldn't be the only termination behaviour, include job completion

    for task in tasks:
        task.cancel()  # TODO: check this behaviour

    await asyncio.gather(*tasks, return_exceptions=True)  # TODO: check this too
    LOGGER.info("Batch driver shutting down")


async def chunk_puller(
    chunk_sem: asyncio.BoundedSemaphore,
    input_chunk_queue: asyncio.Queue[InputChunk],
    shutdown_event: asyncio.Event,
) -> None:
    """Claim chunks up to the local chunk limit without busy waiting."""
    while not shutdown_event.is_set():  # ERROR
        await chunk_sem.acquire()

        try:
            chunk = await get_chunk()
        except Exception:
            chunk_sem.release()
            raise

        if chunk is None:  # Chunk manager says no chunks, but job is ongoing
            chunk_sem.release()
            await asyncio.sleep(NO_CHUNK_BACKOFF_SECONDS)
            continue

        await input_chunk_queue.put(chunk)


async def prompt_driver(
    chunk_sem: asyncio.BoundedSemaphore,
    input_chunk_queue: asyncio.Queue[InputChunk],
    output_chunk_queue: asyncio.Queue[OutputChunk],
    shutdown_event: asyncio.Event,
) -> None:
    """Submit chunk prompts to vLLM and enqueue completed chunks for writing."""
    while not shutdown_event.is_set():  # ERROR
        chunk = await input_chunk_queue.get()

        try:
            completed_chunk = await process_chunk(chunk)
            await output_chunk_queue.put(completed_chunk)
            chunk_sem.release()
        finally:
            input_chunk_queue.task_done()  # For Queue.join()


async def chunk_writer(
    output_chunk_queue: asyncio.Queue[OutputChunk],
    shutdown_event: asyncio.Event,
) -> None:
    """Write completed chunks to external storage."""
    while not shutdown_event.is_set():
        completed_chunk = await output_chunk_queue.get()

        try:
            await put_chunk(completed_chunk)
        finally:
            output_chunk_queue.task_done()  # For Queue.join()


async def get_chunk() -> InputChunk | None:
    """Claim the next chunk from the chunk manager and download it from storage."""
    # TODO
    return None


async def process_chunk(chunk: InputChunk) -> OutputChunk:
    """Submit all prompts in a chunk to vLLM and collect their outputs."""
    # TODO
    return OutputChunk("", [])


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

    asyncio.run(main())
