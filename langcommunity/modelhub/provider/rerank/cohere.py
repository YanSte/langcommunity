import logging
from typing import Any, List, Tuple, Type, Union

from langfoundation.modelhub.rerank.base import BaseRerankModel


logger = logging.getLogger(__name__)


try:
    from cohere.client import Client as CohereClient  # type: ignore[unused-ignore] # noqa: F401
except ImportError:
    raise ImportError()


class RerankCohere(BaseRerankModel):
    """Interface for cross encoder models."""

    client: Union[
        Type[CohereClient],
        Type[Any],
    ]
    model_name: str

    def rerank(
        self,
        query: str,
        docs: List[str],
    ) -> List[Tuple[int, float]]:
        with self.client() as client:
            response = client.rerank(
                model=self.model_name,
                query=query,
                documents=docs,
                top_n=self.top_n,
            )

        return [(result.index, result.relevance_score) for result in response.results]
