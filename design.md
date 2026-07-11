### Overall

The driver processes chunks, where each chunk consists of a number of prompts. The driver will pull chunks from some cloud storage, and chunk assignment is requested from an external chunk manager service. For each chunk, the driver submits individual prompts in the chunk to the vLLM engine and write the output as a complete chunk to the same cloud storage.

The vLLM server is invoked using the CLI `vllm serve`, so that metrics can be exposed. `supervisor.py` is the init script and lifecycle owner, spawning 2 processes - the `vllm serve` engine and `batch_driver.py`, which will drive the vLLM engine.

The Kubernetes deployment is split into 2 cases - A: pipelism parallelism = 1 or B: pipeline parallelism > 1. For case A, the 2 processes are wrapped in 1 container.

### Termination

`supervisor.py` is the lifecycle owner and also PID 1.

Natural termination happens when `batch_driver.py` detects that the job is completed (informed when querying chunk manager service), and exits with 0.

`supervisor.py` detects termination in both the child processes and terminates the other child when one child dies. This is the same behaviour for both natural termination and also unexpected termination, but the exit code will be different.

In K8s, if the pod is terminated, `supervisor.py` receives the SIGTERM and propagates this to both child processes. It generally does not escalate to SIGKILL and leaves that to K8s to force termination. It escalates when it is waiting for a sibling to terminate (and that sibling does not terminate in time), because this case can be self-triggered instead of external triggered by K8s (e.g. job is done, one sibling crashes).
