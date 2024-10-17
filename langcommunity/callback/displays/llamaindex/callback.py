from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from langchain_core.documents import Document as LangchainDocument

from langcommunity.callback.displays.llamaindex.models.record import (
    LLamaIndexRetrieverRecord,
    LLamaIndexRetrieverState,
    LLamaIndexStreamState,
)


logger = logging.getLogger(__name__)

try:
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler  # type: ignore
    from llama_index.core.callbacks.schema import CBEventType, EventPayload  # type: ignore
    from llama_index.core.schema import NodeWithScore  # type: ignore
except ImportError:
    raise ImportError()


class BaseAsyncLLamaIndexCallbackHandler(BaseCallbackHandler, ABC):
    """
    Base callback handler that can be used to track event starts and ends.

    This class is a bridge between the LlamaIndex callback system and the Langchain callback system.
    """

    verbose: bool = True
    record: Optional[LLamaIndexRetrieverRecord] = None
    event_starts_to_ignore: Tuple = tuple()
    event_ends_to_ignore: Tuple = tuple()

    # Abstract
    # ---
    @abstractmethod
    def llama_index_on_retriever(self, retriever: LLamaIndexRetrieverRecord, **kwargs: Any) -> None:
        """
        Abstract method to handle a retriever event.

        Args:
            retriever: The retriever record.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        pass

    @abstractmethod
    async def llama_index_on_stream_token(
        self,
        id: int,
        token: str,
        cumulate_token: str,
        state: LLamaIndexStreamState,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """
        Abstract method to handle a retriever event.

        Args:
            retriever: The retriever record.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        pass

    # Bridge callbacks
    # ---
    async def llama_index_on_prepare(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Callback method called when preparing for LLamaIndex retriever.
        """
        self.record = LLamaIndexRetrieverRecord(
            run_id=run_id,
            parent_run_id=parent_run_id,
            state=LLamaIndexRetrieverState.PREPARED,
            tags=tags,
        )
        self.llama_index_on_retriever(retriever=self.record, **kwargs)

    # Callbacks
    # ---
    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Callback method called when an event starts.
        """
        if event_type != CBEventType.QUERY:
            return

        if not self.record:
            raise ValueError(
                "Retriever record is not initialized. Please Add `on_retriever_prepare_llama_index` to retrieve with managers callback langchaing to initialize the retriever record."  # noqa: E501
            )

        self.record.update(
            query=payload[EventPayload.QUERY_STR.value],
            state=LLamaIndexRetrieverState.START,
        )
        self.llama_index_on_retriever(retriever=self.record, **kwargs)

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Callback method called when an event ends.
        """
        if not self.record:
            raise ValueError(
                "Retriever record is not initialized. Please Add `on_retriever_prepare_llama_index` to "
                "retrieve with managers callback langchaing to initialize the retriever record."
            )

        # NOTE: Take the last one from Retrieve else Rerank
        if event_type == CBEventType.RETRIEVE or event_type == CBEventType.RERANKING:
            # TODO: can be exception
            if payload and EventPayload.NODES in payload:
                documents_nodes: List[NodeWithScore] = payload[EventPayload.NODES]

                documents = []
                for node in documents_nodes:
                    metadata = {**node.metadata}
                    metadata["score"] = node.score
                    documents.append(
                        LangchainDocument(
                            page_content=node.text,
                            metadata=node.metadata,
                        )
                    )
                self.record.update(
                    state=LLamaIndexRetrieverState.END,
                    documents=documents,
                )
            if payload and EventPayload.EXCEPTION in payload:
                logger.error(
                    payload,
                    extra={
                        "title": "[ERROR] Exception in LLAMMA" + " : " + self.__class__.__name__,
                        "verbose": True,
                    },
                )

        elif event_type == CBEventType.QUERY:
            self.llama_index_on_retriever(retriever=self.record, **kwargs)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """
        Callback method called when an overall trace is launched.
        """
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """
        Callback method called when an overall trace is exited.
        """
        pass
