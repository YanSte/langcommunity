from __future__ import annotations

import sys
from typing import Any

from langfoundation.callback.display.records.token import (
    TokenStream,
    TokenStreamState,
)

from langcommunity.callback.displays.mixin.llamaindex.callback import (
    BaseAsyncDisplayMixinCallbackHandler,
)


try:
    from halo import Halo  # type: ignore
except ImportError:
    raise ImportError()


class AsyncCLICallbackHandler(BaseAsyncDisplayMixinCallbackHandler):
    """
    A callback handler for the CLI display that uses the halo library for a spinner.

    This class provides methods to handle various events related to display actions,
    such as LLM (Language Model) events, tool events, retriever events, and agent events.
    """

    spinner: Halo

    def __init__(
        self,
        should_cumulate_token: bool = False,
    ) -> None:
        """
        Initialize the callback handler.

        Args:
            should_cumulate_token: Whether to cumulate tokens.
        """
        self.spinner = Halo(spinner="dots")
        self.spinner.start()
        self.should_cumulate_token = should_cumulate_token

    async def on_token_stream(
        self,
        token: TokenStream,
        **kwargs: Any,
    ) -> None:
        """
        Handle a stream token event.

        Args:
            token: The token to handle.
        """
        if token.state == TokenStreamState.START:
            self.spinner.stop()

        sys.stdout.write(token.token)

        if token.state == TokenStreamState.END:
            sys.stdout.write("\n\n·················································\n\n")
