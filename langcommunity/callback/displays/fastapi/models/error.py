from pydantic import BaseModel


class ErrorStreamResponse(BaseModel):
    """
    Response token for API

    """

    id: int
    error: str
