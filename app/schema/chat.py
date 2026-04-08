from pydantic import BaseModel, Field
from typing import List, Dict, Any, Annotated

class ChatRequest(BaseModel):
    question: str
    history: List[Dict[str, Any]] = []
    model: Annotated[str | None, Field(
        default=None,
        description="(Opcional) O nome do modelo a ser usado (ex: 'gemma-7b-it', 'llama-3.1-8b-instant').",
        example="openai/gpt-oss-120b"
    )]