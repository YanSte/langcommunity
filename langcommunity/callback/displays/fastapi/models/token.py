from pydantic import BaseModel


class TokenStreamResponse(BaseModel):
    """
    Response token for API

    NOTE: Map to Json, that why pydantic v1
    """

    id: int
    token: str
