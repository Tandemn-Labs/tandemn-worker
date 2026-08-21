ARG VLLM_IMAGE=vllm/vllm-openai:v0.22.1
FROM ghcr.io/astral-sh/uv:0.12.5 AS uv
FROM ${VLLM_IMAGE}

WORKDIR /app

# Write logs immediately instead of buffering
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

COPY pyproject.toml ./
COPY --from=uv /uv /usr/local/bin/uv
RUN uv pip install --system --no-cache \
    "googleapis-common-protos" \
    "grpcio>=1.83.0,<2" \
    "grpcio-status>=1.83.0,<2" \
    "protobuf>=7.35.1,<8"

COPY src ./src

# The official vLLM image starts the OpenAI server by default. This worker image
# owns its process entrypoint and can still run `vllm serve` explicitly later.
ENTRYPOINT []
CMD ["python", "-m", "tandemn_worker.supervisor"]
