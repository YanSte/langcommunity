from __future__ import annotations

from typing import Any, List

from langfoundation.callback.base.records.agent import AgentRecord
from langfoundation.callback.base.records.retriever import (
    RetrieverRecord,
)
from langfoundation.callback.base.records.token import (
    TokenStream,
    TokenStreamState,
)
from langfoundation.callback.base.records.tool import ToolRecord

from langcommunity.callback.displays.mixin.llamaindex.callback import (
    BaseAsyncDisplayMixinCallbackHandler,
)


class AsyncDisplayStoreCallbackHandler(BaseAsyncDisplayMixinCallbackHandler):
    """
    A callback handler to store AI messages in a list.
    """

    display_ai_messages: List[str] = []

    def __init__(
        self,
        verbose: bool = False,
    ):
        """
        Initialize the callback handler.

        Args:
            verbose: Whether to print out logs.
        """
        self.display_ai_messages = []
        self.verbose = verbose
        self.should_cumulate_token = True

    def clear(self) -> None:
        """Clear the stored AI messages."""
        self.display_ai_messages = []

    # Stream
    # ----
    async def on_token_stream(
        self,
        token: TokenStream,
        tags: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Handle a token stream event.

        If the token stream is not ended, return immediately.
        """
        if token.state != TokenStreamState.END:
            return

        if tags:
            display_parser = self._find_display_parser_by_tags(tags)
            token_to_append = display_parser.parse(token.cumulate_tokens) if display_parser else token.cumulate_tokens
        else:
            token_to_append = token.cumulate_tokens

        self.display_ai_messages.append(token_to_append)

    # Tool
    # ----
    async def on_tool(self, tool: ToolRecord, **kwargs: Any) -> None:
        """Handle a tool event."""
        pass

    # Agent
    # ----
    async def on_agent(self, agent: AgentRecord, **kwargs: Any) -> None:
        """Handle an agent event."""
        pass

    # Retriever
    # ----
    async def on_retriever(self, retriever: RetrieverRecord, **kwargs: Any) -> None:
        """Handle a retriever event."""
        pass

    def non_async_on_retriever(self, retriever: RetrieverRecord, **kwargs: Any) -> None:
        """Handle a retriever event in a non-async manner."""
        pass

    # Feedback
    # ----
    async def on_feedback(self, feedback: str, **kwargs: Any) -> None:
        """Handle a feedback event."""
        pass
