from sqlalchemy.orm import Session
import app.database.models as models

def get_history_by_session_id(db: Session, session_id: str):
    """Recupera o histórico de uma conversa ordenado por tempo."""
    return db.query(models.ConversationHistory).filter(models.ConversationHistory.session_id == session_id).order_by(models.ConversationHistory.timestamp).all()

def add_message_to_history(db: Session, session_id: str, role: str, content: str):
    """Adiciona uma nova mensagem ao histórico da conversa."""
    db_message = models.ConversationHistory(
        session_id=session_id,
        role=role,
        content=content
    )
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message