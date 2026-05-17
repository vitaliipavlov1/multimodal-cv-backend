"""
Shared async Triton gRPC client — singleton per process.
Thread-safe initialisation via asyncio.Lock.
"""

import asyncio
import logging
from typing import Optional

import tritonclient.grpc.aio as grpcclient

logger = logging.getLogger(__name__)

_client: Optional[grpcclient.InferenceServerClient] = None
_lock   = asyncio.Lock()


async def get_triton_client(url: str = "127.0.0.1:8001") -> grpcclient.InferenceServerClient:
    """Return the shared async Triton client, creating it on first call."""
    global _client
    async with _lock:
        if _client is None:
            _client = grpcclient.InferenceServerClient(url=url, verbose=False)
            logger.info("Triton gRPC client connected: %s", url)
    return _client


async def close_triton_client() -> None:
    """Gracefully close the shared Triton client on application shutdown."""
    global _client
    async with _lock:
        if _client is not None:
            try:
                await _client.close()
                logger.info("Triton gRPC client closed")
            finally:
                _client = None
