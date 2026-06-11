"""Process supervisor for the batch worker container."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys

LOGGER = logging.getLogger(__name__)


def build_vllm_command() -> list[str]:
    """Build the `vllm serve` command from environment variables."""
    model = os.environ.get("VLLM_MODEL", "")
    host = os.environ.get("VLLM_HOST", "0.0.0.0")
    port = os.environ.get("VLLM_PORT", "8000")
    extra_args = shlex.split(os.environ.get("VLLM_EXTRA_ARGS", ""))

    return ["vllm", "serve", model, "--host", host, "--port", port, *extra_args]


def main() -> int:
    """Start vLLM and the batch worker, then wait for the batch worker."""
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

    vllm_command = build_vllm_command()

    LOGGER.info("Starting vLLM server: %s", shlex.join(vllm_command))
    subprocess.Popen(vllm_command)

    worker_command = [sys.executable, "-m", "tandemn_worker.batch_worker"]
    LOGGER.info("Starting batch worker: %s", shlex.join(worker_command))
    worker_process = subprocess.Popen(worker_command)

    return worker_process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
