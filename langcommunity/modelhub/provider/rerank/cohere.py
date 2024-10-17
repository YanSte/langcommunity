import logging
from typing import List, Tuple

from langfoundation.modelhub.rerank.base import BaseRerankModel

try:
    from cohere.client import Client as CohereClient  # type: ignore
    from cohere.client import AsyncClient as AsyncCohereClient  # type: ignore

except ImportError:
    raise ImportError("Please install the `cohere` library by running: `poetry add cohere`.")

logger = logging.getLogger(__name__)


class RerankCohere(BaseRerankModel):
    """Interface for cross encoder models."""

    model_name: str

    def rerank(
        self,
        query: str,
        docs: List[str],
        top_n: int,
    ) -> List[Tuple[int, float]]:
        with CohereClient() as client:
            response = client.rerank(
                model=self.model_name,
                query=query,
                documents=docs,
                top_n=top_n,
            )

        return [(result.index, result.relevance_score) for result in response.results]

    async def arerank(
        self,
        query: str,
        docs: List[str],
        top_n: int,
    ) -> List[Tuple[int, float]]:
        async with AsyncCohereClient() as client:
            response = client.rerank(
                model=self.model_name,
                query=query,
                documents=docs,
                top_n=top_n,
            )

        return [(result.index, result.relevance_score) for result in response.results]
