import logging
from fastapi import FastAPI, Depends, HTTPException, Header
from sqlalchemy.orm import Session

import app.database.crud as crud
import app.database.models as models
from app.database.database import SessionLocal, engine

from app.chatbot import get_chatbot_response
from app.schema.chat import ChatRequest

models.Base.metadata.create_all(bind=engine)

logging.basicConfig(level=logging.INFO)
app = FastAPI(
    title="Chatbot API Juridico",
    version="2.1.0"
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

try:
    with open("data/context.txt", "r", encoding="utf-8") as f:
        BACKGROUND_CONTEXT = f.read()
except FileNotFoundError:
    logging.error("CRÍTICO: Arquivo 'context.txt' não encontrado. A API não pode funcionar.")
    BACKGROUND_CONTEXT = None


@app.post("/chat/")
def handle_chat_request(
    request: ChatRequest,
    session_id: str = Header(..., description="ID único para a sessão da conversa."),
    db: Session = Depends(get_db)
):
    if not BACKGROUND_CONTEXT:
        raise HTTPException(status_code=500, detail="Erro interno: Contexto do chatbot não configurado.")

    chosen_model = request.model if request.model else "openai/gpt-oss-120b"
    logging.info(f"Usando o modelo: {chosen_model}")

    db_history = crud.get_history_by_session_id(db, session_id=session_id)
    formatted_history = [{"role": msg.role, "content": msg.content} for msg in db_history]

    response_text = get_chatbot_response(
        BACKGROUND_CONTEXT,
        request.question,
        formatted_history,
        model=chosen_model
    )
    
    crud.add_message_to_history(db, session_id=session_id, role="user", content=request.question)
    crud.add_message_to_history(db, session_id=session_id, role="assistant", content=response_text)

    return {"response": response_text}