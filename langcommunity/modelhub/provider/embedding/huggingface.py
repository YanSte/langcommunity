from typing import Any, Dict, Type

from langchain_core.embeddings import Embeddings
from langfoundation.modelhub.embedding.base import BaseEmbeddingProvider


try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except ImportError:
    raise ImportError()


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    GTE_LARGE_EN_V_1_5 = "Alibaba-NLP/gte-large-en-v1.5"

    @property
    def provider(self) -> Type[Embeddings]:
        return HuggingFaceEmbeddings

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "model_name": self.value,
            "model_kwargs": {"trust_remote_code": True},
        }
