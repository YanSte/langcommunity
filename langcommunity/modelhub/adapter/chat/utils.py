from typing import Dict, List, Sequence

try:
    from llama_index.core.base.llms.types import ChatMessage, LLMMetadata, MessageRole  # type: ignore[unused-ignore] # noqa: F401
    from llama_index.llms.openai.utils import openai_modelname_to_contextsize  # type: ignore[unused-ignore] # noqa: F401
except ImportError:
    raise ImportError("Please install `llama_index` library. Please install `poetry add 'langcommunity[llama_index]'`.")


DEFAULT_ANYSCALE_API_BASE = "https://api.endpoints.anyscale.com/v1"
DEFAULT_ANYSCALE_API_VERSION = ""

LLAMA_MODELS = {
    "meta-llama/Llama-2-7b-chat-hf": 4096,
    "meta-llama/Llama-2-13b-chat-hf": 4096,
    "meta-llama/Llama-2-70b-chat-hf": 4096,
    "codellama/CodeLlama-34b-Instruct-hf": 16384,
    "Meta-Llama/Llama-Guard-7b": 4096,
}

MISTRAL_MODELS = {
    "mistralai/Mistral-7B-Instruct-v0.1": 16384,
    "Open-Orca/Mistral-7B-OpenOrca": 8192,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 32768,
}

ZEPHYR_MODELS = {
    "HuggingFaceH4/zephyr-7b-beta": 16384,
}

ALL_AVAILABLE_MODELS = {
    **LLAMA_MODELS,
    **MISTRAL_MODELS,
    **ZEPHYR_MODELS,
}

DISCONTINUED_MODELS: Dict[str, int] = {}


def anyscale_modelname_to_contextsize(modelname: str) -> int:
    """
    Calculate the maximum number of tokens possible to generate for a model.

    Args:
        modelname: The modelname we want to know the context size for.

    Returns:
        The maximum context size

    Example:
        .. code-block:: python

            max_tokens = anyscale_modelname_to_contextsize(model_name)
    """
    # handling finetuned models
    # TO BE FILLED

    if modelname in DISCONTINUED_MODELS:
        raise ValueError(f"Anyscale hosted model {modelname} has been discontinued. " "Please choose another model.")

    context_size = ALL_AVAILABLE_MODELS.get(modelname, None)

    if context_size is None:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid Anyscale model name."
            "Known models are: " + ", ".join(ALL_AVAILABLE_MODELS.keys())
        )

    return context_size


class LC:
    from llama_index.core.bridge.langchain import (  # AI21,; Cohere,; OpenAI,
        AIMessage,
        BaseChatModel,
        BaseLanguageModel,
        BaseMessage,
        ChatAnyscale,
        ChatFireworks,
        ChatMessage,
        ChatOpenAI,
        FunctionMessage,
        HumanMessage,
        SystemMessage,
    )


def is_chat_model(llm: LC.BaseLanguageModel) -> bool:
    return isinstance(llm, LC.BaseChatModel)


def to_lc_messages(messages: Sequence[ChatMessage]) -> List[LC.BaseMessage]:
    lc_messages: List[LC.BaseMessage] = []
    for message in messages:
        LC_MessageClass = LC.BaseMessage
        lc_kw = {
            "content": message.content,
            "additional_kwargs": message.additional_kwargs,
        }
        if message.role == "user":
            LC_MessageClass = LC.HumanMessage
        elif message.role == "assistant":
            LC_MessageClass = LC.AIMessage
        elif message.role == "function":
            LC_MessageClass = LC.FunctionMessage
        elif message.role == "system":
            LC_MessageClass = LC.SystemMessage
        elif message.role == "chatbot":
            LC_MessageClass = LC.ChatMessage
        else:
            raise ValueError(f"Invalid role: {message.role}")

        for req_key in LC_MessageClass.schema().get("required"):
            if req_key not in lc_kw:
                more_kw = lc_kw.get("additional_kwargs")
                if not isinstance(more_kw, dict):
                    raise ValueError(f"additional_kwargs must be a dict, got {type(more_kw)}")
                if req_key not in more_kw:
                    raise ValueError(f"{req_key} needed for {LC_MessageClass}")
                lc_kw[req_key] = more_kw.pop(req_key)

        lc_messages.append(LC_MessageClass(**lc_kw))

    return lc_messages


def from_lc_messages(lc_messages: Sequence[LC.BaseMessage]) -> List[ChatMessage]:
    messages: List[ChatMessage] = []
    for lc_message in lc_messages:
        li_kw = {
            "content": lc_message.content,
            "additional_kwargs": lc_message.additional_kwargs,
        }
        if isinstance(lc_message, LC.HumanMessage):
            li_kw["role"] = MessageRole.USER
        elif isinstance(lc_message, LC.AIMessage):
            li_kw["role"] = MessageRole.ASSISTANT
        elif isinstance(lc_message, LC.FunctionMessage):
            li_kw["role"] = MessageRole.FUNCTION
        elif isinstance(lc_message, LC.SystemMessage):
            li_kw["role"] = MessageRole.SYSTEM
        elif isinstance(lc_message, LC.ChatMessage):
            li_kw["role"] = MessageRole.CHATBOT
        else:
            raise ValueError(f"Invalid message type: {type(lc_message)}")
        messages.append(ChatMessage(**li_kw))

    return messages


def get_llm_metadata(llm: LC.BaseLanguageModel) -> LLMMetadata:
    """Get LLM metadata from llm."""
    if not isinstance(llm, LC.BaseLanguageModel):
        raise ValueError("llm must be instance of LangChain BaseLanguageModel")

    is_chat_model_ = is_chat_model(llm)

    # if isinstance(llm, LC.OpenAI):
    #     return LLMMetadata(
    #         context_window=openai_modelname_to_contextsize(llm.model_name),
    #         num_output=llm.max_tokens,
    #         is_chat_model=is_chat_model_,
    #         model_name=llm.model_name,
    #     )
    if isinstance(llm, LC.ChatAnyscale):
        return LLMMetadata(
            context_window=anyscale_modelname_to_contextsize(llm.model_name),
            num_output=llm.max_tokens or -1,
            is_chat_model=is_chat_model_,
            model_name=llm.model_name,
        )
    # elif isinstance(llm, LC.ChatFireworks):
    #     return LLMMetadata(
    #         context_window=fireworks_modelname_to_contextsize(llm.model_name),
    #         num_output=llm.max_tokens or -1,
    #         is_chat_model=is_chat_model_,
    #         model_name=llm.model_name,
    #     )
    elif isinstance(llm, LC.ChatOpenAI):
        return LLMMetadata(
            context_window=openai_modelname_to_contextsize(llm.model_name),
            num_output=llm.max_tokens or -1,
            is_chat_model=is_chat_model_,
            model_name=llm.model_name,
        )
    # elif isinstance(llm, LC.Cohere):
    #     # June 2023: Cohere's supported max input size for Generation models is 2048
    #     # Reference: <https://docs.cohere.com/docs/tokens>
    #     return LLMMetadata(
    #         context_window=COHERE_CONTEXT_WINDOW,
    #         num_output=llm.max_tokens,
    #         is_chat_model=is_chat_model_,
    #         model_name=llm.model,
    #     )
    # elif isinstance(llm, LC.AI21):
    #     # June 2023:
    #     #   AI21's supported max input size for
    #     #   J2 models is 8K (8192 tokens to be exact)
    #     # Reference: <https://docs.ai21.com/changelog/increased-context-length-for-j2-foundation-models>
    #     return LLMMetadata(
    #         context_window=AI21_J2_CONTEXT_WINDOW,
    #         num_output=llm.maxTokens,
    #         is_chat_model=is_chat_model_,
    #         model_name=llm.model,
    #     )
    else:
        return LLMMetadata(is_chat_model=is_chat_model_)
