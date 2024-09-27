from enum import auto
from typing import List, Optional
from uuid import UUID

from langchain_core.documents import Document as LangchainDocument
from langfoundation.callback.display.records.base import BaseRecord
from strenum import LowercaseStrEnum


class LLamaIndexRetrieverState(LowercaseStrEnum):
    """
    Enum class representing the possible states of a LLamaIndex retriever record.
    """

    PREPARED = auto()
    START = auto()
    END = auto()


class LLamaIndexStreamState(LowercaseStrEnum):
    """
    Enum class representing the possible states of a stream record for a LLamaIndex retriever.
    """

    START = auto()
    STREAM = auto()
    END = auto()


class LLamaIndexRetrieverRecord(BaseRecord):
    """
    Represents a retriever record.

    Attributes:
        run_id (UUID): The ID of the run.
        parent_run_id (Optional[UUID], optional): The ID of the parent run. Defaults to None.
        query (str): The query string.
        state (RetrieverState): The state of the retriever.
        documents (Optional[List[Document]], optional): The list of documents. Defaults to None.
    """

    run_id: Optional[UUID] = None
    parent_run_id: Optional[UUID] = None
    tags: List[str] | None = None

    query: Optional[str] = None
    state: LLamaIndexRetrieverState

    documents: Optional[List[LangchainDocument]] = None
