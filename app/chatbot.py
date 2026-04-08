import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
from app.sanitization import sanitize_response

# Configura o logging para exibir informações úteis no terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Pega a chave da API da Groq a partir das variáveis de ambiente
api_key = os.getenv("GROQ_API_KEY")

# Validação para garantir que a chave da API foi encontrada
if not api_key:
    logging.error("A variável de ambiente GROQ_API_KEY não foi encontrada!")
    raise ValueError("A chave da API da Groq precisa ser configurada no arquivo .env.")

# Inicializa o cliente, apontando para a URL da API da Groq
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key,
)

def get_chatbot_response(context: str, question: str, history: list, model: str = "llama3-8b-8192") -> str:
    """
    Gera uma resposta de um LLM usando um contexto, uma pergunta e o histórico da conversa.

    Args:
        context: O texto base que o LLM deve usar para responder.
        question: A pergunta atual do usuário.
        history: Uma lista de dicionários representando as trocas anteriores (user/assistant).
        model: O nome do modelo a ser usado (padrão: "mixtral-8x7b-32768").

    Returns:
        A resposta gerada pelo modelo.
    """
    system_prompt = f"""
    # Sua Identidade e Objetivo
    Você é um assistente de IA da JurisConsultoria. Seu objetivo é ser prestativo, profissional e responder às perguntas dos usuários baseando-se **exclusivamente** no "CONTEÚDO DE REFERÊNCIA" fornecido abaixo. Sua função é facilitar o acesso à informação, não dar conselhos.

    ## Suas Diretrizes de Conversa
    Para cada pergunta do usuário, siga estes passos:

    1.  **Análise Cuidadosa:** Leia a pergunta e verifique se a resposta pode ser encontrada no "CONTEÚDO DE REFERÊNCIA".
    
    2.  **Elaboração da Resposta:**
        * **Se a informação estiver disponível:** Responda à pergunta de forma clara e amigável. Você pode reescrever e resumir a informação para que a conversa soe natural, mas não adicione nada que não esteja no texto original.
        * **Se a informação NÃO estiver disponível:** Se a resposta não estiver no conteúdo, ou se o usuário pedir um conselho legal, uma opinião ou ajuda com um caso específico, você deve educadamente recusar e direcioná-lo para um especialista.

    ## Modelo de Resposta para Recusa
    Quando precisar recusar, use uma frase parecida com esta, adaptando-a levemente se necessário para o contexto da conversa:

    "Não tenho informações sobre este tópico. Para uma análise detalhada do seu caso ou para outras dúvidas, o ideal é agendar uma consulta inicial com um de nossos advogados especialistas. Você pode fazer isso em nosso site (www.jurisconsultoria.com.br/agendamento) ou pelo WhatsApp (11) 99876-5432)."

    ## O que Evitar:
    - **Não invente:** Nunca use conhecimento externo ou faça suposições. Se não está no texto, você não sabe.
    - **Não peça dados:** Não solicite informações que não sejam estritamente necessárias para entender a pergunta dentro do contexto da conversa atual.
    - **Não dê conselhos:** Lembre-se, você é um assistente informativo, não um advogado. Se um usuário pedir um conselho, use o modelo de recusa.

    ## CONTEÚDO DE REFERÊNCIA (Sua Única Fonte de Verdade)
    ---
    {context}
    ---
    """

    # Monta a lista de mensagens completa para a API
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    messages.extend(history)  # Adiciona o histórico da conversa
    messages.append({"role": "user", "content": question})  # Adiciona a pergunta atual

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
        raw_response = response.choices[0].message.content
        sanitized_response = sanitize_response(raw_response)
        return sanitized_response

    except Exception as e:
        logging.error(f"Ocorreu um erro ao chamar a API da Groq: {e}")
        return "Desculpe, não consegui processar sua pergunta no momento."

# Bloco para teste direto do arquivo (não é executado quando importado pelo main.py)
if __name__ == '__main__':
    # Simula o carregamento do contexto a partir de um arquivo
    try:
        with open("data/context.txt", "r", encoding="utf-8") as f:
            test_context = f.read()
    except FileNotFoundError:
        print("Erro: Arquivo 'context.txt' não encontrado para o teste.")
        test_context = ""

    if test_context:
        print("Iniciando teste local do chatbot...")
        
        # Teste 1: Primeira pergunta
        q1 = "Qual o valor da consulta inicial?"
        h1 = [] # Histórico vazio
        answer1 = get_chatbot_response(test_context, q1, h1)
        print(f"\nPergunta 1: {q1}")
        print(f"Resposta 1: {answer1}")

        # Teste 2: Pergunta de acompanhamento com histórico
        q2 = "E esse valor é abatido?"
        h2 = [
            {"role": "user", "content": q1},
            {"role": "assistant", "content": answer1}
        ]
        answer2 = get_chatbot_response(test_context, q2, h2)
        print(f"\nPergunta 2: {q2}")
        print(f"Resposta 2: {answer2}")