from datetime import datetime
from functools import cache
from typing import Optional

from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.tools import BaseTool, StructuredTool, Tool
from pydantic import BaseModel, Field

from langcommunity.tools.python_ast_repl_tool import PythonAstREPLTool
from langcommunity.tools.you import YahooFinanceStockPriceTool

###################################
# Tools
###################################
# TODO: For tool inject use a special << >>  to indicate format the value


def get_yahoo_finance_news_tool() -> StructuredTool:
    """Look up things on wikipedia."""

    class YahooInput(BaseModel):
        """Input for the yahoo_finance_news tool."""

        query: str = Field(
            description="Input query should be only one ticker symbol separate or enter multiple tickers separated by a comma. Example: 'AAPL, MSFT, NVDA' or 'AAPL'."  # noqa: E501
        )

    @cache
    def run(
        query: str,
    ) -> str:
        return YahooFinanceStockPriceTool().run(query)

    return StructuredTool.from_function(
        func=run,
        name="YahooFinanceStockPrice",
        description="Useful for obtaining stock price information, this tool accepts ticker symbol such as 'AAPL' for Apple, 'MSF' for Microsoft, or 'NVDA' for NVIDIA.",  # noqa: E501
        args_schema=YahooInput,
    )


def get_duckduckgo_search_tool() -> StructuredTool:
    class DuckDuckGoSearchInput(BaseModel):
        """Look up things online."""

        query: str = Field(description="Input should be a search query.")

    def handle_error(error) -> str:
        return "The following errors occurred during tool execution:" + error.args[0] + "Please try another tool."

    @cache
    def run(
        query: str,
    ) -> str:
        return DuckDuckGoSearchRun(handle_tool_error=handle_error).run(query)

    return StructuredTool.from_function(
        func=run,
        name="DuckDuckGoSearch",
        description="Useful for when you need to answer questions about current events",
        args_schema=DuckDuckGoSearchInput,
    )


def get_wiki_tool(verbose: bool = False) -> StructuredTool:
    """Look up things on wikipedia."""

    class WikiInput(BaseModel):
        """Input for the yahoo_finance_news tool."""

        query: str = Field(description="Input should be a search query. Examples: 'Macron' or 'History of France'")

    @cache
    def run(
        query: str,
    ) -> str:
        return WikipediaQueryRun(
            description=(
                "Useful for when you need to answer general questions about "
                "people, places, companies, facts, historical events, or other subjects."
            ),
            api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2048),  # type: ignore
            args_schema=WikiInput,
        ).run(query)

    return StructuredTool.from_function(
        func=run,
        name="Wikipedia",
        description=(
            "Useful for when you need to answer general questions about "
            "people, places, companies, facts, historical events, or other subjects."
        ),
        args_schema=WikiInput,
    )


def get_date_time_tool() -> BaseTool:
    def get_date(nothing) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M")

    return Tool(
        name="Date-time",
        func=get_date,
        description=("Useful to ask the current date and time." "Not Input needed."),
    )


###################################
# Code
###################################


###################################
# Python REPL
###################################


class PythonInputs(BaseModel):
    query: str = Field(prompt="Code snippet to run")


def get_python_tool(description: Optional[str] = None) -> StructuredTool:
    """Look up things on wikipedia."""

    @cache
    def run(
        query: str,
    ) -> str:
        class PythonREPL:
            def __init__(self):
                self.local_vars = {}
                self.python_tool = PythonAstREPLTool()

            def run(self, code: str) -> str:
                output = str(self.python_tool.run(code))
                if output == "":
                    return "Executed successfully."
                else:
                    return output

        python_repl = PythonREPL()
        return python_repl.run(query)

    default_prompt = (
        "A Python shell.\n"
        "This is a Python shell interface designed for executing Python commands.\n"
        "Ensure that your input adheres to proper Python syntax.\n"
        "Make sure it does not look abbreviated before using it in your answer.\n"
        "Python executed will return:\n"
        "- If successful, the result of the executed code or msg success if result of the executed code is empty.\n"
        "- Otherwise, it will display the Python error encountered."
    )
    return StructuredTool.from_function(
        func=run,
        name="PythonShell",
        description=description or default_prompt,
        args_schema=PythonInputs,
    )
