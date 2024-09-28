from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Union

from langfoundation.callback.base.records.token import (
    TokenStream,
    TokenStreamState,
)

from langcommunity.callback.displays.fastapi.models.complete import (
    CompleteStreamResponse,
)
from langcommunity.callback.displays.fastapi.models.error import ErrorStreamResponse
from langcommunity.callback.displays.fastapi.models.token import TokenStreamResponse
from langcommunity.callback.displays.mixin.llamaindex.callback import (
    BaseAsyncDisplayMixinCallbackHandler,
)


class AsyncFastAPICallbackHandler(BaseAsyncDisplayMixinCallbackHandler):
    """Callback handler that returns an async iterator.

    This class implements the callbacks for the FastAPI integration.
    """

    queue: asyncio.Queue[Union[TokenStreamResponse, CompleteStreamResponse, ErrorStreamResponse]]

    def __init__(
        self,
        should_cumulate_token: bool = False,
    ) -> None:
        """
        Args:
            should_cumulate_token (bool): Whether to cumulate the token. Defaults to False.
        """
        self.should_cumulate_token = should_cumulate_token
        self.queue = asyncio.Queue()

    # Callbacks
    # ----
    async def on_token_stream(
        self,
        token: TokenStream,
        **kwargs: Any,
    ) -> None:
        match token.state:
            case TokenStreamState.ERROR:
                self.queue.put_nowait(
                    ErrorStreamResponse(
                        id=token.id,
                        error=str(token.error) if token.error else "error",
                    )
                )
            case _:
                if token.state == TokenStreamState.END:
                    self.queue.put_nowait(
                        CompleteStreamResponse(
                            id=token.id,
                        )
                    )
                else:
                    self.queue.put_nowait(
                        TokenStreamResponse(
                            id=token.id,
                            token=token.token,
                        )
                    )

    # Token Iterator
    # ----
    async def atoken_iterator(self) -> AsyncIterator[str]:
        """Returns an async iterator over the token stream."""
        while True:
            reponse = await self.queue.get()
            if isinstance(reponse, CompleteStreamResponse):
                yield "data: " + "[DONE]"
                self.queue.task_done()
                break
            elif isinstance(reponse, ErrorStreamResponse):
                yield "data: " + reponse.json()
                yield "data: " + "[DONE]"
                self.queue.task_done()
                break
            else:
                yield "data: " + reponse.json()
