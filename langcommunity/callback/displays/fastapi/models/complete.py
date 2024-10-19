from pydantic import BaseModel


class CompleteStreamResponse(BaseModel):
    """
    Response token for API
    """

    id: int
