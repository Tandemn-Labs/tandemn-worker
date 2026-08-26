# Code structure / File hierarchy

### Chunk manager gRPC clients

The gRPC/protobuf schema is here:
```text
proto/tandemn/chunkmanager/v1/chunk_manager.proto
```
Manually update the file for now, at some point, we can do some Git submodule or version control for it to match the version sitting in the `chunk-manager` repo.

Run the following command from the repository root:
```sh
uv run --extra dev python -m grpc_tools.protoc \
  -I proto \
  --python_out=src \
  --pyi_out=src \
  --grpc_python_out=src \
  proto/tandemn/chunkmanager/v1/chunk_manager.proto
```

This generates:

```text
# Protobuf messages, enums, descriptors, and serialization code.
src/tandemn/chunkmanager/v1/chunk_manager_pb2.py

# Type declarations for type checkers and IDEs (Optional but helpful)
src/tandemn/chunkmanager/v1/chunk_manager_pb2.pyi

# Generated gRPC client stubs and server registration helpers.
src/tandemn/chunkmanager/v1/chunk_manager_pb2_grpc.py
```

The `__init__.py` files are not generated  (manually added).
