import logging
from functools import lru_cache
from typing import Any, List, Tuple, Type, Union

from langchain_community.cross_encoders import BaseCrossEncoder
from langfoundation.modelhub.rerank.base import BaseRerankModel, BaseRerankProvider

try:
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
except ImportError:
    raise ImportError()


logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _get_cross_encoder_instance(
    model_name: str,
    provider: Union[
        Type[BaseCrossEncoder],
        Type[Any],
    ],
) -> BaseCrossEncoder:
    config = {
        "model_name": model_name,
        "model_kwargs": {
            "trust_remote_code": True,
        },
    }
    logger.info(
        config,
        extra={
            "title": "[Cache Model] " + " : " + model_name,
            "verbose": True,
        },
    )
    return provider(**config)


class RerankCrossEncoder(BaseRerankModel):
    """Interface for cross encoder models."""

    model_name: str
    provider: Union[
        Type[BaseCrossEncoder],
        Type[Any],
    ]

    @property
    def _model(
        self,
    ) -> BaseCrossEncoder:
        return _get_cross_encoder_instance(
            model_name=self.model_name,
            provider=self.provider,
        )

    def rerank(
        self,
        query: str,
        docs: List[str],
    ) -> List[Tuple[int, float]]:
        query_and_nodes = [
            (
                query,
                doc,
            )
            for doc in docs
        ]
        scores = self._model.score(query_and_nodes)

        indexs_scores = [(index, score) for index, score in enumerate(scores)]

        return sorted(indexs_scores, key=lambda x: -x[1] if x[1] else 0)[: self.top_n]


class HuggingFaceRerankProvider(BaseRerankProvider):
    BGE_RERANKER_LAGER = "BAAI/bge-reranker-large"

    @property
    def provider(
        self,
    ) -> Union[
        Type[BaseCrossEncoder],
        Type[Any],
    ]:
        return HuggingFaceCrossEncoder

    def model(
        self,
        top_n: int,
    ) -> BaseRerankModel:
        return RerankCrossEncoder(
            top_n=top_n,
            model_name=self.value,
            provider=self.provider,
        )
