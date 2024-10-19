from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.summary import SummarizerMixin
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import BaseMessage, get_buffer_string
from pydantic import Field


class ConversationSummaryChatMemory(BaseChatMemory, SummarizerMixin):
    """Buffer with summarizer for storing conversation memory."""

    trigger_summary_token_limit: int = Field(
        description="Maximum number of tokens in the buffer before triggering summarization.",
        default=1000,
    )
    history_max_summary_buffer_token_limit: int = Field(
        description="Maximum number of tokens take from history for summarization buffer.",
        default=2000,
    )
    summary_last_messages_include_limit: int = Field(
        description="Number of most recent messages to include in the summary after pruning.",
        default=2,
    )

    memory_key: str = Field(
        description="Key used for accessing the memory buffer.",
        default="history",
    )

    chat_summurized: BaseChatMessageHistory = Field(
        description="History of chat messages that have been summarized.",
        default_factory=InMemoryChatMessageHistory,
    )

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    @property
    def summury_or_history_messages(self) -> List[BaseMessage]:
        """String buffer of memory."""
        if self.chat_summurized.messages:
            return self.chat_summurized.messages
        else:
            return self.chat_memory.messages

    @property
    def summury_message(self) -> Optional[BaseMessage]:
        """String buffer of memory."""
        if self.chat_summurized.messages:
            return self.chat_summurized.messages[0]
        else:
            return None

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        buffer = self.summury_or_history_messages
        if self.return_messages:
            final_buffer: Any = buffer
        else:
            final_buffer = get_buffer_string(buffer, human_prefix=self.human_prefix, ai_prefix=self.ai_prefix)
        return {self.memory_key: final_buffer}

    async def aload_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Asynchronously return key-value pairs given the text input to the chain."""
        raise NotImplementedError()

    def summarize(self) -> None:
        """Summarize the chat history"""
        buffer = self.chat_memory.messages
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.trigger_summary_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.history_max_summary_buffer_token_limit:
                pruned_memory.append(buffer.pop(0))
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
            else:
                pruned_memory = buffer

            moving_summary_buffer: str = cast(str, self.summury_message.content if self.summury_message else "")
            moving_summary_buffer = self.predict_new_summary(pruned_memory, moving_summary_buffer)
            first_messages: List[BaseMessage] = [self.summary_message_cls(content=moving_summary_buffer)]

            buffer = first_messages + buffer[-self.summary_last_messages_include_limit :]
            self.clear_summarized()
            self.add_summarized_messages(buffer)

    def add_summarized_messages(
        self,
        messages: List[BaseMessage],
    ) -> None:
        """Append a list of messages to summarize"""
        self.chat_summurized.add_messages(messages)

    def clear_summarized(self) -> None:
        """Clear a list of summarized messages"""
        self.chat_summurized.clear()

    def clear(self) -> None:
        self.clear_summarized()
        self.chat_memory.clear()
