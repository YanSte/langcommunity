from typing import Iterable, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pandas import DataFrame
from requests.exceptions import HTTPError, ReadTimeout
from urllib3.exceptions import ConnectionError


class YahooFinanceStockPriceTool(BaseTool):
    """Tool that searches financial news on Yahoo Finance."""

    name: str = "yahoo_finance_news"
    description: str = (
        "Useful for when you need to find financial news "
        "about a public company. "
        "Input should be a company ticker. "
        "For example, AAPL for Apple, MSFT for Microsoft."
    )
    top_k: int = 10
    """The number of results to return."""

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Yahoo Finance News tool."""
        try:
            import yfinance
        except ImportError:
            raise ImportError("Could not import yfinance python package. " "Please install it with `pip install yfinance`.")
        company = yfinance.Ticker(query)
        try:
            if company.isin is None:
                return f"Company ticker {query} not found."
        except (HTTPError, ReadTimeout, ConnectionError):
            return f"Company ticker {query} not found."

        data: DataFrame = yfinance.download(query, period="5d")
        return data.to_string()

    @staticmethod
    def _format_results(docs: Iterable[Document], query: str) -> str:
        doc_strings = [
            "\n".join([doc.metadata["title"], doc.metadata["description"]])
            for doc in docs
            if query in doc.metadata["description"] or query in doc.metadata["title"]
        ]
        return "\n\n".join(doc_strings)
