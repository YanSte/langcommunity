from pydantic import BaseModel


class ErrorStreamResponse(BaseModel):
    """
    Response token for API

    NOTE: Map to Json, that why pydantic v1
    """

    id: int
    error: str
