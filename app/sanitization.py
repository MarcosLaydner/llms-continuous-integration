import re

EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

def sanitize_response(text: str) -> str:
    """
    Limpa a resposta do LLM, removendo caracteres indesejados e espaços extras.
    """
    cleaned_text = text.strip()
    cleaned_text = cleaned_text.replace('"', '')
    cleaned_text = ' '.join(cleaned_text.split())
    
    cleaned_text = EMAIL.sub("[REDACTED_EMAIL]", cleaned_text)
    return cleaned_text