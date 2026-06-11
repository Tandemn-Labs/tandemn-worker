ARG VLLM_IMAGE=vllm/vllm-openai:v0.22.1
FROM ${VLLM_IMAGE}

WORKDIR /app

# Write logs immediately instead of buffering
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

COPY pyproject.toml ./
COPY src ./src

# The official vLLM image starts the OpenAI server by default. This worker image
# owns its process entrypoint and can still run `vllm serve` explicitly later.
ENTRYPOINT []
CMD ["python", "-m", "tandemn_worker.supervisor"]
