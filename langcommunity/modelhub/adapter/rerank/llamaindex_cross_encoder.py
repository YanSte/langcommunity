import logging
from typing import List, Optional

from langfoundation.modelhub.rerank.config import RerankConfig


logger = logging.getLogger(__name__)

try:
    from llama_index.core.callbacks import CBEventType, EventPayload  # type: ignore[unused-ignore] # noqa: F401
    from llama_index.core.postprocessor.types import BaseNodePostprocessor  # type: ignore[unused-ignore] # noqa: F401
    from llama_index.core.schema import NodeWithScore, QueryBundle  # type: ignore[unused-ignore] # noqa: F401
except ImportError:
    raise ImportError()


class LlamaIndexCrossEncoder(BaseNodePostprocessor):
    config: RerankConfig
    keep_retrieval_score: bool = True
    keep_rerank_score: bool = True

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        query = query_bundle.query_str
        # TODO: To test LLM, embedding 🚜
        docs = [node.get_text() for node in nodes]
        model = self.config.model()
        top_n = self.config.top_n

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.config.model_provider,
                EventPayload.QUERY_STR: query,
                EventPayload.TOP_K: self.config.top_n,
            },
        ) as event:
            indexes_scores = model.rerank(query=query, docs=docs)
            # Incase response not sorted and not remove top_n
            indexes_scores = sorted(indexes_scores, key=lambda x: x[1], reverse=True)[:top_n]

            sorted_nodes = []
            for index, score in indexes_scores:
                node = nodes[index]

                if self.keep_retrieval_score:
                    node.node.metadata["retrieval_score"] = nodes[index].score
                if self.keep_rerank_score:
                    node.node.metadata["rerank_score"] = score

                sorted_nodes.append(node)
            event.on_end(payload={EventPayload.NODES: sorted_nodes})

        return sorted_nodes

    @classmethod
    def class_name(cls) -> str:
        return "SentenceTransformerRerank"
