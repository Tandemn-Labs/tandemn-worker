"""Process supervisor for the batch driver container."""

from __future__ import annotations

import logging
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time

LOGGER = logging.getLogger(__name__)
ChildExit = tuple[str, int]  # E.g. ("batch driver", 0)
SIGTERM_TIMEOUT_SECONDS = 30.0  # Does not always escalate SIGTERM to SIGKILL
SIGKILL_TIMEOUT_SECONDS = 5.0
DRIVER_GRACE_TIMEOUT_SECONDS = float(
    os.getenv("TD_DRIVER_GRACE_TIMEOUT_SECONDS", str(SIGTERM_TIMEOUT_SECONDS))
)


# TODO: Figure out env vars for vLLM
def build_vllm_command() -> list[str]:
    """Build the `vllm serve` command from environment variables."""
    executable = os.environ.get("TD_VLLM_EXECUTABLE", "vllm")
    model = os.environ.get("TD_VLLM_MODEL", "")
    host = os.environ.get("TD_VLLM_HOST", "0.0.0.0")
    port = os.environ.get("TD_VLLM_PORT", "8000")
    extra_args = shlex.split(os.environ.get("TD_VLLM_EXTRA_ARGS", ""))

    return [executable, "serve", model, "--host", host, "--port", port, *extra_args]


def signal_process(
    process: subprocess.Popen[bytes],
    name: str,
    signum: signal.Signals,
) -> None:
    """Send a signal to a child process group if it is still running."""
    log_level = logging.WARNING if signum == signal.SIGKILL else logging.INFO
    LOGGER.log(log_level, "Sending %s to %s (pid=%s)", signum.name, name, process.pid)
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        return


def watch_process(
    process: subprocess.Popen[bytes],
    name: str,
    events: queue.Queue[ChildExit],
) -> threading.Thread:
    """Returns a daemon thread that reports when given child process exits, reports by writing to queue."""

    def wait_and_report() -> None:
        return_code = process.wait()
        if return_code < 0:
            return_code = 128 + abs(return_code)
        LOGGER.info("%s exited with return code %s", name, return_code)
        events.put((name, return_code))

    thread = threading.Thread(target=wait_and_report, name=f"watch-{name}", daemon=True)
    thread.start()
    return thread


def wait_for_remaining_child(
    events: queue.Queue[ChildExit],
    remaining_process: subprocess.Popen[bytes],
    remaining_name: str,
    timeout_seconds: float = SIGTERM_TIMEOUT_SECONDS,
) -> None:
    """Wait briefly for the remaining child, then force kill it."""
    try:
        proc_name, _ = events.get(timeout=timeout_seconds)
        if proc_name == remaining_name:
            return
        else:
            LOGGER.warning(
                "Expected %s to terminate, but received termination event for %s",
                remaining_name,
                proc_name,
            )
            signal_process(remaining_process, remaining_name, signal.SIGKILL)
    except queue.Empty:
        LOGGER.warning(
            "%s did not exit within %.1f seconds; force terminating",
            remaining_name,
            timeout_seconds,
        )
        signal_process(remaining_process, remaining_name, signal.SIGKILL)

    try:
        remaining_process.wait(timeout=SIGKILL_TIMEOUT_SECONDS)
    except (subprocess.SubprocessError, OSError) as exc:
        LOGGER.error(
            "%s did not exit cleanly within %.1f seconds after SIGKILL: %s",
            remaining_name,
            SIGKILL_TIMEOUT_SECONDS,
            exc,
        )


def main() -> int:
    """Start vLLM and the batch driver, then supervise their lifecycle."""

    # Spawns the 2 child processes
    # start_new_session calls the setsid syscall in the child
    vllm_command = build_vllm_command()
    LOGGER.info("Starting vLLM server: %s", shlex.join(vllm_command))
    vllm_proc = subprocess.Popen(vllm_command, start_new_session=True)

    driver_command = [sys.executable, "-m", "tandemn_worker.batch_driver"]
    LOGGER.info("Starting batch driver: %s", shlex.join(driver_command))
    # Could fail, but don't have to explicitly terminate vLLM since this runs in K8s
    driver_proc = subprocess.Popen(driver_command, start_new_session=True)

    # Spawn watcher threads that wait on child termination
    child_events: queue.Queue[ChildExit] = queue.Queue()
    watch_process(vllm_proc, "vLLM server", child_events)
    watch_process(driver_proc, "batch driver", child_events)

    # Signal handler + attachment to SIGINT and SIGTERM
    shutdown_signal: int | None = None  # Prevent duplicate handling + Indicator

    def handle_shutdown(signum: int, _frame: object) -> None:
        nonlocal shutdown_signal
        if shutdown_signal is None:
            LOGGER.info("Received signal %s; gracefully terminating batch driver", signum)
            shutdown_signal = signum
            signal_process(driver_proc, "batch driver", signal.SIGTERM)

            def force_kill_driver_after_grace() -> None:
                time.sleep(DRIVER_GRACE_TIMEOUT_SECONDS)
                if driver_proc.poll() is None:
                    LOGGER.warning(
                        "Batch driver did not exit within %.1f seconds; force terminating",
                        DRIVER_GRACE_TIMEOUT_SECONDS,
                    )
                    signal_process(driver_proc, "batch driver", signal.SIGKILL)

            threading.Thread(
                target=force_kill_driver_after_grace,
                name="force-kill-batch-driver",
                daemon=True,
            ).start()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Main blocking call that waits on either child to terminate
    child_name, return_code = child_events.get()

    # External termination signal received + handled (propagate to children)
    if shutdown_signal is not None:
        if child_name == "batch driver":
            signal_process(vllm_proc, "vLLM server", signal.SIGTERM)
            wait_for_remaining_child(child_events, vllm_proc, "vLLM server")
        else:
            LOGGER.error(
                "vLLM server exited before batch driver drained; stopping driver immediately"
            )
            signal_process(driver_proc, "batch driver", signal.SIGUSR1)
            wait_for_remaining_child(
                child_events,
                driver_proc,
                "batch driver",
            )
        return 128 + shutdown_signal

    # Below cases occur if either child terminates without external signal

    # Case 1: If batch driver is the one that terminates first
    if child_name == "batch driver":
        if return_code == 0:
            LOGGER.info("Batch driver completed naturally; terminating vLLM server")
        else:
            LOGGER.error(
                "Batch driver exited with code %s; terminating vLLM server",
                return_code,
            )

        signal_process(vllm_proc, "vLLM server", signal.SIGTERM)
        wait_for_remaining_child(child_events, vllm_proc, "vLLM server")
        return return_code

    # Case 2: vLLM terminates unexpectedly
    LOGGER.error(
        "vLLM server exited unexpectedly with code %s; terminating batch driver",
        return_code,
    )
    signal_process(driver_proc, "batch driver", signal.SIGUSR1)
    wait_for_remaining_child(child_events, driver_proc, "batch driver")
    return return_code or 1


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

    raise SystemExit(main())
