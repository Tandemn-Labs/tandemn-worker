### Overall

The driver processes chunks, where each chunk consists of a number of prompts. The driver will pull chunks from some cloud storage, and chunk assignment is requested from an external chunk manager service. For each chunk, the driver submits individual prompts in the chunk to the vLLM engine and write the output as a complete chunk to the same cloud storage.

The vLLM server is invoked using the CLI `vllm serve`, so that metrics can be exposed. `supervisor.py` is the init script and lifecycle owner, spawning 2 processes - the `vllm serve` engine and `batch_driver.py`, which will drive the vLLM engine.

The Kubernetes deployment is split into 2 cases - A: pipelism parallelism = 1 or B: pipeline parallelism > 1. For case A, the 2 processes are wrapped in 1 container.

### Chunking

The number of input chunks that can be held locally is controlled by a semaphore (which can be set by an envvar). The semaphore is acquired when an input chunk is successfully downloaded and released when `prompt_driver` writes a complete chunk to the output chunk queue (not when an output chunk is successfully written!). A slot for downloading an input chunk should not be taken up by an output chunk being written.

For now, the limit on output chunks is 3 * limit on input chunks. This can help in setting some backpressure so that we don't have excessive input chunk downloading + processing while the output chunks pile up with slow write to external storage. The backpressure occurs when `prompt_driver` tries to put in another completed output chunk but the output chunk queue is full. However, in general, we want the GPUs to stay busy, so the output chunk queue has a higher limit than the input chunk queue limit.


### Termination

`supervisor.py` is the lifecycle owner and also PID 1.

Natural termination happens when `batch_driver.py` detects that the job is completed (informed when querying chunk manager service), and exits with 0.

`supervisor.py` detects termination in both the child processes and terminates the other child when one child dies. This is the same behaviour for both natural termination and also unexpected termination, but the exit code will be different.

In K8s, if the pod is terminated, `supervisor.py` receives the SIGTERM and propagates this to both child processes. It generally does not escalate to SIGKILL and leaves that to K8s to force termination. It escalates when it is waiting for a sibling to terminate (and that sibling does not terminate in time), because this case can be self-triggered instead of external triggered by K8s (e.g. job is done, one sibling crashes).

### Prompt processing

`prompt_driver() -> submit_prompt() -> submit_prompt_request()`

#### `prompt_driver()`
The main task/coroutine that is actually coordinating the work, it pulls out prompts from the active chunk, saturates the vLLM engine and receives output as they come in (out of order).

#### `submit_prompt()`
Really just a wrapper to check if the lease is stale. It creates 2 tasks - (1) `chunk.lease.state_event.wait()` and (2) `submit_prompt_request()`. (1) waits for the stale `asyncio.Event` to be set, while (2) actually does the HTTP request/response. If stale event happens first, prompt gets cancelled.

#### `submit_prompt_request()`
Actual HTTP request construction happens here. The returned `PromptResult` has an `output` field which is basically a JSON. This JSON contains a response payload and an error payload. If there is an error (HTTP timeout, HTTP error), it sets response payloade to None and populates the error payload. Vice versa for the successful case. Either case, the payloads are wrapped up in `PromptResult` and returned.

#### Error propagation
`submit_prompt_request()` catches all exceptions and just fold it into the `PromptResult.output` string, which is basically a JSON dump with the 2 key fields of `response` and `error`.



### Metrics

The batched metrics are exposed on an `/metrics` endpoint. They have the following key stats: total input chunks pulled, total output chunks written, num reqs inflight, num reqs processed.
