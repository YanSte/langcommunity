from __future__ import annotations

from typing import Any, Callable, Optional

from langfoundation.callback.base.records.agent import (
    AgentRecord,
    AgentState,
)
from langfoundation.callback.base.records.retriever import (
    RetrieverRecord,
    RetrieverState,
)
from langfoundation.callback.base.records.token import (
    TokenStream,
    TokenStreamState,
)
from langfoundation.callback.base.records.tool import (
    ToolRecord,
    ToolsState,
)

from langcommunity.callback.displays.mixin.llamaindex.callback import BaseAsyncDisplayMixinCallbackHandler
from langcommunity.utils.debug.formatter import Formatter


try:
    import streamlit as st  # type: ignore
    from streamlit.delta_generator import DeltaGenerator  # type: ignore
except ImportError:
    raise ImportError()


class AsyncStreamlitCallbackHandler(BaseAsyncDisplayMixinCallbackHandler):
    def __init__(
        self,
        msg_container: DeltaGenerator,
        chain_thought_container: Optional[DeltaGenerator] = None,
        on_start_stream_callback: Optional[Callable[..., None]] = None,
        max_tool_input_str_length: int = 60,
        max_document_str_length: int = 400,
        separator="- - -",
        verbose: bool = True,
    ):
        self._msg_container = msg_container
        self._chain_thought_container = chain_thought_container
        self._on_output_start = on_start_stream_callback

        self._max_tool_input_str_length = max_tool_input_str_length
        self._max_document_str_length = max_document_str_length
        self._separator = separator

        self.verbose = verbose
        self.should_cumulate_token = True

        self._llm_token_stream_placeholder = None
        self._status = None
        self._display_token = ""

    # Stream
    # ----
    async def on_token_stream(
        self,
        token: TokenStream,
        **kwargs: Any,
    ) -> None:
        with self._msg_container:
            if self._on_output_start:
                self._on_output_start()
                self._on_output_start = None

            if self._llm_token_stream_placeholder is None:
                self._llm_token_stream_placeholder = st.empty()

            self._display_token += token.token
            spacer_indicator = "" if token.state == TokenStreamState.END else "▌"
            display_text = self._display_token + spacer_indicator
            self._llm_token_stream_placeholder.markdown(display_text)

    # Tool
    # ----
    async def on_tool(self, tool: ToolRecord, **kwargs: Any) -> None:
        if not self._chain_thought_container:
            return

        with self._status.container():
            if tool.state == ToolsState.REQUEST:
                self._status.update(label=f"**Tool Request:** {tool.name}", state="running")
                st.markdown(f"**Tool Request:**{Formatter.format(tool.input_str)}\n{self._separator}")

            if tool.state == ToolsState.START:
                self._status.update(label=f"**Tool Start:** {tool.name}", state="running")
                st.markdown(f"**Tool Input:** {Formatter.format(tool.input_str)}\n{self._separator}")

            elif tool.state == ToolsState.END:
                self._status.update(label=f"**Tool End:** {tool.name}", state="running")
                st.markdown(f"**Tool Ouput:** {Formatter.format(tool.output)}\n{self._separator}")

            elif tool.state == ToolsState.ERROR:
                self._status.update(label=f"**Tool Error:** {tool.name}", expanded=True, state="error")
                st.exception(tool.error)
                st.markdown(f"**Tool Error:** {Formatter.format(tool.error)}\n{self._separator}")

    async def on_agent(self, agent: AgentRecord, **kwargs: Any) -> None:
        if not self._chain_thought_container:
            return

        if agent.state == AgentState.THOUGHT:
            with self._chain_thought_container:
                self._status = st.status(label="**Agent Start**", state="running")
                with self._status.container():
                    lines = agent.log.strip().split("\n")
                    result = []
                    for line in lines:
                        try:
                            key, value = line.strip().split(":")
                            result.append(f"**Agent {key}:** {value}")
                        except Exception:
                            result.append(f"{line}")
                    result_text = "\n\n".join(result)
                    st.markdown(f"{result_text}\n{self._separator}")

        elif agent.state == AgentState.END:
            # NOTE: Special Agent, status can be null when Angent answer directly without tools
            status = self._status if self._status else st.status(label="**Agent Start**", state="running")
            with status.container():
                status.update(label="**Agent**", state="complete")
                st.markdown(f"**Agent Result:** {Formatter.format(agent.result)}")
            self._status = None

    async def on_retriever(self, retriever: RetrieverRecord, **kwargs: Any) -> None:
        self._dislay_on_retriever(retriever)

    def non_async_on_retriever(self, retriever: RetrieverRecord, **kwargs: Any) -> None:
        self._dislay_on_retriever(retriever)

    async def on_feedback(self, feedback: str, **kwargs: Any) -> None:
        with self._chain_thought_container:
            with st.status(label="**Feedback**", state="complete").container():
                st.markdown(feedback)

    def _dislay_on_retriever(
        self,
        retriever: RetrieverRecord,
    ) -> None:
        if retriever.state == RetrieverState.START:
            with self._chain_thought_container:
                self._status = st.status(label="**Retrieve**", state="running")

        elif retriever.state == RetrieverState.END:
            doc_samples = "\n".join(
                [
                    f"**Chunk {index}:** "
                    + document.page_content[: self._max_document_str_length]
                    + "..."
                    + "\n\n"
                    + "**Meta Data:**\n"
                    + f"```\n\n{Formatter.format(document.metadata)}\n\n```"
                    + "\n\n"
                    + "----"
                    for index, document in enumerate(retriever.documents)
                ]
            )
            with self._status.container():
                self._status.update(state="complete")
                st.markdown(doc_samples)

            self._status = None


def _convert_newlines(text: str) -> str:
    """Convert newline characters to markdown newline sequences
    (space, space, newline).
    """
    return text.replace("\n", "  \n")
