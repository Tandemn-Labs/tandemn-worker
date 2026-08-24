# AGENTS.md

This file is for coding agents. It is laid out as organization wide rules followed by repo-specific information.

Current repo: tandemn-labs/tandemn-worker

## Organization guide

### Overall coding style
- Avoid clever one-liners that hurt readability.
- Use comments only for non-obvious operational logic, failure modes, or cross-service contracts. Do not comment what the code already says.
- Follow the existing local patterns before inventing a new one.
- Simplicity first. No features beyond what was asked. No abstractions for single-use code. No "flexibility" or "configurability" that wasn't requested. No error handling for impossible scenarios. Do not add unnecessary complexity in order to attain goals like scalability and security.
- Make only surgical changes. Touch only what is needed, don't improve or refractor anything that is not absolutely necessary.
- Work backwards; Define the GOAL first (success criteria) then ASK QUESTIONS till verified. Your goal is to transform the goal into sub-tasks and verifiable goals. For multi-step tasks, state a brief plan.
- Don't remove existing explanatory comments unless the changes to the code changed the validity of the comments

### Python rules
- Use PEP 8 as code style guide and PEP 257 as docstrings style guide.
- Ensure `pyproject.toml` exists with `ruff`, `mypy` rules
- Use the `./src/` layout for code
- Use `uv` for virtual environment
- Use the python stdlib `logging` library instead of `print()` in the codebase

### Testing Philosophy
- Integration tests should use local containers; never real cloud accounts.

### Repository Boundaries
- Do not commit credentials, .env files, generated caches, local Docker volumes, or large artifacts.

### YAML rules
- Use `.yaml` for new files.

### Other rules
- Ensure `.pre-commit-config.yaml` exists


## Repo-specific guide

This repository contains the code for the vLLM batched inference worker that will run as jobs in Kubernetes. The driver processes chunks, where each chunk consists of a number of prompts. The driver will pull chunks from some cloud storage, and chunk assignment is requested from an external chunk manager service. For each chunk, the driver submits individual prompts in the chunk to the vLLM engine and write the output as a complete chunk to the same cloud storage.

The vLLM server is invoked using the CLI `vllm serve`, so that metrics can be exposed. `supervisor.py` is the init script and lifecycle owner, spawning 2 processes - the `vllm serve` engine and `batch_driver.py`, which will drive the vLLM engine.

The Kubernetes deployment is split into 2 cases - A: pipelism parallelism = 1 or B: pipeline parallelism > 1. For case A, the 2 processes are wrapped in 1 container.
