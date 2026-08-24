"""Prometheus metrics endpoint and driver metric handles."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    generate_latest,
)

from tandemn_worker.config import MetricsConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DriverMetrics:
    """Prometheus metrics updated by the batch driver."""

    inflight_requests: Gauge
    requests_processed: Counter
    input_chunks_pulled: Counter
    output_chunks_written: Counter


def create_metrics_app(config: MetricsConfig) -> tuple[FastAPI, DriverMetrics]:
    """Create the FastAPI app exposing driver metrics in Prometheus format."""
    registry = CollectorRegistry()

    inflight_requests = Gauge(
        "batched_reqs_inflight",
        "Number of prompt requests currently in flight.",
        registry=registry,
    )

    requests_processed = Counter(
        "batched_reqs_processed",
        "Total number of prompt requests that have received a response.",
        registry=registry,
    )

    input_chunks_pulled = Counter(
        "batched_chunks_input_pulled",
        "Total number of input chunks pulled.",
        registry=registry,
    )

    output_chunks_written = Counter(
        "batched_chunks_output_written",
        "Total number of output chunks written.",
        registry=registry,
    )

    metrics = DriverMetrics(
        inflight_requests=inflight_requests,
        requests_processed=requests_processed,
        input_chunks_pulled=input_chunks_pulled,
        output_chunks_written=output_chunks_written,
    )

    app = FastAPI()
    metrics_path = f"/{config.path.strip('/')}" if config.path.strip("/") else "/metrics"

    @app.get(metrics_path, include_in_schema=False)
    async def metrics_endpoint() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app, metrics


def start_metrics_server(
    app: FastAPI,
    config: MetricsConfig,
) -> tuple[uvicorn.Server, asyncio.Task[None]]:
    """Start the metrics ASGI server in the current event loop."""
    server_config = uvicorn.Config(
        app,
        host=config.host,
        port=config.port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(server_config)
    task = asyncio.create_task(server.serve(), name="metrics-server")
    LOGGER.info(
        "Batched metrics endpoint listening on http://%s:%s%s",
        config.host,
        config.port,
        config.path,
    )
    return server, task


async def stop_metrics_server(server: uvicorn.Server, task: asyncio.Task[None]) -> None:
    """Request a graceful metrics server shutdown."""
    server.should_exit = True
    await asyncio.gather(task, return_exceptions=True)
