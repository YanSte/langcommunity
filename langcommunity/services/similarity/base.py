from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseSimilarity(ABC):
    @abstractmethod
    def calculate_similarity(self, query: str, documents: List[str]) -> List[float]:
        """
        Calculate the cosine similarity between the query and a list of documents.

        Parameters:
        - query (str): The query sentence.
        - documents (List[str]): List of document sentences.

        Returns:
        - List[float]: List of similarity scores between the query and each document.
        """

    @abstractmethod
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
