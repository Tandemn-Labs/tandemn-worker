ARG VLLM_IMAGE=vllm/vllm-openai:v0.28.0@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14
FROM ghcr.io/astral-sh/uv:0.12.5 AS uv
FROM ${VLLM_IMAGE}

WORKDIR /app

# Write logs immediately instead of buffering
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    TD_VLLM_EXECUTABLE=/usr/local/bin/vllm

COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN UV_PROJECT_ENVIRONMENT=/opt/tandemn-worker-venv \
    uv sync --locked --no-dev --no-install-project --no-cache \
    --python /usr/bin/python3 --no-python-downloads

COPY src ./src

# The official vLLM image starts the OpenAI server by default. This worker image
# owns its process entrypoint and can still run `vllm serve` explicitly later.
# Keep the worker venv off PATH so the vLLM executable retains its base-image Python.
ENTRYPOINT []
CMD ["/opt/tandemn-worker-venv/bin/python", "-m", "tandemn_worker.supervisor"]
