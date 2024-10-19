from typing import Any, Dict, Type

from langchain_core.embeddings import Embeddings
from langfoundation.modelhub.embedding.base import BaseEmbeddingProvider

try:
    from modelhub.adapter.langchain_overrides.mistral_embeddings import MistralAIEmbeddings  # type: ignore
except ImportError:
    raise ImportError()


class MistralEmbeddingProvider(BaseEmbeddingProvider):
    EMBED = "mistral-embed"

    @property
    def provider(self) -> Type[Embeddings]:
        return MistralAIEmbeddings

    @property
    def config(self) -> Dict[str, Any]:
        return {"model": self.value}
