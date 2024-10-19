from typing import List, Tuple, Union

import torch
from gary.services.similarity.base import BaseSimilarity
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim, semantic_search


class Similarity(BaseSimilarity):
    """
    # Similarity

    This class provides functionality for computing semantic similarity between a query sentence and a list of documents.
    It utilizes pre-trained models from the Hugging Face Model Hub, specifically designed for sentence similarity tasks.

    The default model used is 'sentence-transformers/sentence-t5-base', which has demonstrated good performance in various tasks.

    Other available models can be explored on the Hugging Face website under the 'sentence-similarity' pipeline tag:
    https://huggingface.co/models?pipeline_tag=sentence-similarity&sort=downloads

    Some notable models include:
    - 'sentence-transformers/all-MiniLM-L12-v2'
    - 'sentence-transformers/paraphrase-MiniLM-L6-v2'
    - sentence-transformers/all-mpnet-base-v2 (400mb)
    - sentence-transformers/sentence-t5-base (120mb)
    - BAAI/bge-large-zh-v1.5 (https://huggingface.co/BAAI/bge-large-en-v1.5) (1.32Gb) (Good result)
    - mixedbread-ai/mxbai-embed-large-v1
    - etc.

    Usage:
    ```
    similarity = Similarity(sentence_transformers_name="sentence-transformers/sentence-t5-base")
    query = "Ma maman est belle"
    documents = ["Ma maman est partie", "Ma maman est magnifique"]
    similarity_results = similarity.semantic_search(query, documents)
    similarity_scores = similarity.calculate_similarity(query, documents)
    ```
    """

    def __init__(self, sentence_transformers_name: str = "mixedbread-ai/mxbai-embed-large-v1") -> None:
        """
        Initialize the Similarity class.

        Parameters:
        - sentence_transformers_name (str): The name or path of the pre-trained sentence transformers model.
                                                Default is "sentence-transformers/sentence-t5-base".
        """
        self._device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._sentence_transformers_name: str = sentence_transformers_name
        self._embeddings_model: Union[SentenceTransformer, None] = None

    # Public Methods
    # ---

    async def configure(self) -> None:
        """
        Initialize the sentence embeddings model.
        """
        if self._embeddings_model is None:
            self._embeddings_model = await self.load_model()

        self._embeddings_model.to(self._device)
        self._embeddings_model.eval()

    async def load_model(self) -> SentenceTransformer:
        return SentenceTransformer(self._sentence_transformers_name, cache_folder="./cache")

    def calculate_similarity(self, query: str, documents: List[str]) -> List[float]:
        """
        Calculate the cosine similarity between the query and a list of documents.

        Parameters:
        - query (str): The query sentence.
        - documents (List[str]): List of document sentences.

        Returns:
        - List[float]: List of similarity scores between the query and each document.
        """
        # Encode sentences
        query_embedding, documents_embeddings = self._encode_query_documents(query, documents)

        # Calculate cosine similarity
        return [cos_sim(query_embedding, emb).item() for emb in documents_embeddings]

    def semantic_search(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Perform semantic search between the query and a list of documents.

        Parameters:
        - query (str): The query sentence.
        - documents (List[str]): List of document sentences.
        - top_k (int): The number of top results to return. Default is 5.

        Returns:
        - List[Tuple[int, float]]: A list of tuples containing the index of the document
        and its similarity score with the query.
        """
        # Encode the query and documents
        query_embedding, documents_embeddings = self._encode_query_documents(query, documents)

        # Perform semantic search
        search_results = semantic_search(query_embedding, documents_embeddings, top_k=top_k)

        # Extract indices and scores from search results
        results = [(result["corpus_id"], result["score"]) for result in search_results[0]]
        return results

    # Private Methods
    # ---

    def _encode(self, sentence: str) -> torch.Tensor:
        """
        Encode a sentence into its embedding representation.

        Parameters:
        - sentence: The input sentence.

        Returns:
        - torch.Tensor: The embedding representation of the input sentence.
        """
        if self._embeddings_model is None:
            raise ValueError("Embeddings model not initialized. Please call setup() first.")
        with torch.no_grad():
            sentence_embedding = self._embeddings_model.encode(sentence, convert_to_tensor=True)
        return sentence_embedding.to(self._device)

    def _encode_query_documents(self, query: str, documents: List[str]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encode the query and a list of documents into their respective embeddings.

        Parameters:
        - query (str): The query sentence.
        - documents (List[str]): List of document sentences.

        Returns:
        - Tuple[torch.Tensor, List[torch.Tensor]]: A tuple containing the embedding of the query
                                                    and a list of embeddings for the documents.
        """
        # Encode the query and documents
        query_embedding = self._encode(query)
        documents_embeddings = [self._encode(doc) for doc in documents]
        return query_embedding, documents_embeddings
