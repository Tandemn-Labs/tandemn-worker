# tandemn-worker

A vLLM batch worker that claims JSONL chunks from a Tandemn chunk manager.

## Run locally

Requires Linux, Python 3.11 or newer, and a running chunk manager with an
existing job, rank, chain, and S3-backed chunk assignments.

Create separate environments for the worker and vLLM so their Python
dependencies do not conflict:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e .

uv venv --python 3.12 ".venv-vllm"
uv pip install --python ".venv-vllm/bin/python" "vllm==0.28.0"
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
export AWS_DEFAULT_REGION="us-east-1"

python -m tandemn_worker.supervisor
```

The supervisor starts both `vllm serve` and the batch driver. Chunk references
must be S3 URIs with this layout:

```text
s3://<bucket>/<prefix>/<job_id>/input/<chunk_id>.jsonl
```

For each lease generation, the worker publishes the corresponding output to:

```text
s3://<bucket>/<prefix>/<job_id>/output/<chunk_id>/<generation>.jsonl
```

The worker uses the standard AWS SDK credential chain, including environment
credentials, shared profiles, EC2 roles, EKS IRSA, and EKS Pod Identity. Its
AWS identity needs `s3:GetObject` on input and output objects and `s3:PutObject`
on output objects. Output read access is used to verify idempotent publication.

Worker metrics are available at `http://localhost:9000/metrics`. In the
container, `TD_VLLM_EXECUTABLE` is set to `/usr/local/bin/vllm` from the base
image.


## Build the container

The image is based on the official `vllm/vllm-openai:v0.28.0` image. vLLM
runs from the base image's system Python, while the worker and all of its
dependencies run from `/opt/tandemn-worker-venv`. The worker environment is
not added to `PATH`, so invoking vLLM cannot accidentally select the worker's
Python interpreter.

Build the image for `linux/amd64`:

```bash
docker build \
  --platform linux/amd64 \
  --tag tandemn-worker:v0.0.1-vllm0.28.0 \
  .
```

## Publish to Google Artifact Registry

The image is already built locally. Push it to Google Artifact Registry

```bash
export IMAGE="us-docker.pkg.dev/tandemn/tandemn-worker/tandemn-worker:v0.0.1-vllm0.28.0"

gcloud auth configure-docker us-docker.pkg.dev

docker tag tandemn-worker:v0.0.1-vllm0.28.0 "${IMAGE}"
docker push "${IMAGE}"
```


## Run on GKE

`deploy/gke/job.yaml` is a example template.

The template optionally imports from a Secret named `tandemn-worker-secrets`. Put the AWS envvars for S3 access in here.

```bash
kubectl --namespace tandemn-system create secret generic tandemn-worker-secrets \
  --from-literal=AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
  --from-literal=AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}"
```

Change the other envvars accordingly
