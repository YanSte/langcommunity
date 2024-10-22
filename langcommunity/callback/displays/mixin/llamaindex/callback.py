from __future__ import annotations

from abc import ABC
from typing import Any

from langfoundation.callback.base.base import (
    BaseAsyncDisplayCallbackHandler,
)
from langfoundation.callback.base.records.agent import AgentRecord
from langfoundation.callback.base.records.retriever import (
    RetrieverRecord,
)
from langfoundation.callback.base.records.token import (
    TokenStreamState,
)
from langfoundation.callback.base.records.tool import ToolRecord

from langcommunity.callback.displays.llamaindex.callback import (
    BaseAsyncLLamaIndexCallbackHandler,
)
from langcommunity.callback.displays.llamaindex.models.record import (
    LLamaIndexRetrieverRecord,
    LLamaIndexRetrieverState,
    LLamaIndexStreamState,
)


class BaseAsyncDisplayMixinCallbackHandler(BaseAsyncDisplayCallbackHandler, BaseAsyncLLamaIndexCallbackHandler, ABC):
    """
    A callback handler for asynchronous base display with mixed functionality.

    This class extends from Non Langchain to Langchain CallbackHandler.

    It provides a bridge between non-Langchain (LLama Index) and Langchain.
    """

    # Bridge LLama Index to Langchain
    # ----

    def llama_index_on_retriever(self, retriever: LLamaIndexRetrieverRecord, **kwargs: Any) -> None:
        """
        Handles the retrieval of LLama index.

        Bridge LLama Index to Langchain
        """
        if retriever.state == LLamaIndexRetrieverState.START:
            self.non_async_on_retriever_start(
                query=retriever.query,
                run_id=retriever.run_id,
                parent_run_id=retriever.parent_run_id,
                tags=retriever.tags,
                **kwargs,
            )
        elif retriever.state == LLamaIndexRetrieverState.END:
            self.non_async_on_retriever_end(
                documents=retriever.documents,
                tags=retriever.tags,
                **kwargs,
            )

    async def on_tool(self, tool: ToolRecord, **kwargs: Any) -> None:
        pass

    async def on_agent(self, agent: AgentRecord, **kwargs: Any) -> None:
        pass

    async def on_retriever(self, retriever: RetrieverRecord, **kwargs: Any) -> None:
        pass

    def non_async_on_retriever(self, retriever: RetrieverRecord, **kwargs: Any) -> None:
        pass

    async def on_feedback(self, feedback: str, **kwargs: Any) -> None:
        pass

    def _to_stream_token(self, state: LLamaIndexStreamState) -> TokenStreamState:
        if state == LLamaIndexStreamState.STREAM:
            return TokenStreamState.STREAM
        elif state == LLamaIndexStreamState.START:
            return TokenStreamState.START
        elif state == LLamaIndexStreamState.END:
            return TokenStreamState.END
