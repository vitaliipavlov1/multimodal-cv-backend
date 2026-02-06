import asyncio
import tritonclient.grpc.aio as grpcclient

_triton_client = None
_client_lock = asyncio.Lock()

async def get_triton_client(url: str = "127.0.0.1:8001"):
    """
    Возвращает асинхронный Triton client.
    Потокобезопасная инициализация.
    """
    global _triton_client
    async with _client_lock:
        if _triton_client is None:
            _triton_client = grpcclient.InferenceServerClient(
                url=url,
                verbose=False
            )
    return _triton_client


async def close_triton_client():
    """
    Корректно закрывает Triton client при shutdown приложения.
    """
    global _triton_client

    async with _client_lock:
        if _triton_client is not None:
            try:
                await _triton_client.close()
            finally:
                _triton_client = None
