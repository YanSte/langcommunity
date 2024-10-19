from pydantic import BaseModel


class TokenStreamResponse(BaseModel):
    """
    Response token for API

    """

    id: int
    token: str
