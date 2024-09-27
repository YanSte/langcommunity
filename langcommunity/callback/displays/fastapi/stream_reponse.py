from __future__ import annotations

import asyncio
import logging
import typing
from typing import AsyncIterable, Coroutine

from langcommunity.callback.displays.fastapi.callback import AsyncFastAPICallbackHandler


logger = logging.getLogger(__name__)

try:
    from fastapi.responses import StreamingResponse as FastApiStreamingResponse  # type: ignore
    from starlette.background import BackgroundTask  # type: ignore
except ImportError:
    raise ImportError()


class CallbackStreamResponse(FastApiStreamingResponse):
    """
    A special response that run an invoke and send a callback response to the client.

    This class is a wrapper around FastAPI's StreamingResponse. It will run the invoke
    and send the callback response to the client as an event stream.
    """

    def __init__(
        self,
        invoke: Coroutine,
        callback: AsyncFastAPICallbackHandler,
        status_code: int = 200,
        headers: typing.Mapping[str, str] | None = None,
        media_type: str | None = "text/event-stream",
        background: BackgroundTask | None = None,
    ) -> None:
        """
        Initialize the response.

        Args:
        invoke: The invoke to run.
        callback: The callback to send the response to.
        status_code: The status code of the response.
        headers: The headers of the response.
        media_type: The media type of the response.
        background: The background task to run.
        """
        super().__init__(
            content=CallbackStreamResponse.send_message(callback, invoke),
            status_code=status_code,
            headers=headers,
            media_type=media_type,
            background=background,
        )

    @staticmethod
    async def send_message(callback: AsyncFastAPICallbackHandler, invoke: Coroutine) -> AsyncIterable[str]:
        """
        Send the message to the callback.

        Args:
        callback: The callback to send the message to.
        invoke: The invoke to run.

        Yields:
        The message to send to the client.
        """
        asyncio.create_task(invoke)

        async for token in callback.atoken_iterator():
            yield token
