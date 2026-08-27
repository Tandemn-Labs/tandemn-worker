# tandemn-worker

A vLLM batch worker that claims JSONL chunks from a Tandemn chunk manager.

## Run locally

Requires Linux, Python 3.11 or newer, and a running chunk manager with an
existing job, rank, chain, and chunk assignments.

Create separate environments for the worker and vLLM so their Python
dependencies do not conflict:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e .

uv venv --python 3.12 ".venv-vllm"
uv pip install --python ".venv-vllm/bin/python" vllm
```

From the repository root, set the worker identity and model, then start the
supervisor:

```bash
source .venv/bin/activate
export PYTHONPATH=src
export TD_VLLM_EXECUTABLE=".venv-vllm/bin/vllm"
export TD_VLLM_MODEL="Qwen/Qwen2-0.5B"
export TD_CHUNK_MANAGER_ADDRESS="CHUNK_MANAGER_HOST:PORT"
export TD_JOB_ID="JOB_ID"
export TD_RANK_ID="RANK_ID"
export TD_CHAIN_ID="0"

python -m tandemn_worker.supervisor
```

The supervisor starts both `vllm serve` and the batch driver. Chunk references
must currently be local paths or `file://` URLs visible to the worker. Worker
metrics are available at `http://localhost:9000/metrics`. In the container,
`TD_VLLM_EXECUTABLE` can remain unset because it defaults to `vllm` from the
base image's `PATH`.
