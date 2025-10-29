"""
Backward-compatible entry point for the async inference policy server.

Rather than running a custom FastAPI stub, the remote policy now relies on the
shared async inference gRPC implementation. You can start the server from this
module or via ``python -m lerobot.async_inference.policy_server``.
"""

from lerobot.async_inference.policy_server import serve

if __name__ == "__main__":
    serve()
