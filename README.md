# llms-continuous-integration
Repository for learning continuous llms integrations with deepeval mlflow and github actions

# Projeto: Avaliação Contínua de LLMs com GitHub Actions, Deepeval e FastAPI

Bem-vindo(a) ao repositório do curso Alura "LLMOps: Avaliação Contínua de LLMs com GitHub Actions e Deepeval". Este projeto demonstra como construir um pipeline de CI/CD para garantir a qualidade e segurança de aplicações baseadas em LLMs, com foco em automação, testes e boas práticas de DevOps.

## 🚀 Sobre o Projeto

O projeto consiste em uma API de chatbot desenvolvida em FastAPI, que utiliza modelos LLM hospedados via Groq/OpenRouter para responder perguntas com base em um contexto jurídico. O pipeline automatizado executa uma suíte de testes especializada com o **Deepeval** para validar métricas como `Faithfulness`, `AnswerRelevancy`, `Bias` e `Toxicity`.

## ✨ Tecnologias Utilizadas

- **Python 3.11+**
- **FastAPI**: Framework web para a API do chatbot
- **Uvicorn**: Servidor ASGI para rodar a aplicação FastAPI
- **OpenAI SDK**: Integração com modelos LLM via Groq/OpenRouter
- **Deepeval**: Framework de avaliação de LLMs
- **Pytest**: Framework de testes
- **SQLAlchemy**: ORM para persistência do histórico de conversas (SQLite)
- **Docker**: Containerização da aplicação
- **GitHub Actions**: Pipeline CI/CD automatizado
- **python-dotenv**: Gerenciamento de variáveis de ambiente

## ⚙️ Configuração do Ambiente Local

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/SEU-USUARIO/NOME-DO-REPOSITORIO.git
   cd NOME-DO-REPOSITORIO
   ```

2. **Crie e ative um ambiente virtual (alternativamente pode ser usado anaconda):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Unix/macOS
   # .\venv\Scripts\activate  # Windows
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure o arquivo `.env`:**
   Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:
   ```env
   GROQ_API_KEY="sua-chave-groq"
   OPENROUTER_API_KEY="sua-chave-openrouter"
   CONFIDENT_API_KEY="sua-chave-deepeval"
   ```

## ▶️ Executando o Projeto Localmente

1. **Construa e execute a API FastAPI com Docker:**
   ```bash
   docker build -t chatbot-api .
   docker run --env-file .env -p 8000:8000 --name my-chatbot-api -v "$(pwd)":/code chatbot-api
   ```
   Acesse a documentação interativa em [http://localhost:8000/docs](http://localhost:8000/docs).

   **OU Execute loclamnte no ambiente virtual usando:**
      ```
      uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
      ```

2. **Execute os testes e avaliações Deepeval:**
   ```bash
   deepeval test run tests/test_chatbot_flow.py -vv
   ```

## 🤖 Pipeline CI/CD com GitHub Actions

O pipeline definido em [`.github/workflows/evaluation.yml`](.github/workflows/evaluation.yml) executa:

- Build e execução do container Docker
- Health check da API
- Execução dos testes Deepeval
- (Opcional) Deploy automatizado para Google Cloud

**Para funcionar em seu fork, configure os segredos do repositório:**
- `GROQ_API_KEY`
- `OPENROUTER_API_KEY`
- `CONFIDENT_API_KEY`

## 📚 Referências

- [FastAPI](https://fastapi.tiangolo.com/)
- [Deepeval](https://deepeval.com/)
- [Groq](https://groq.com/)
- [OpenRouter](https://openrouter.ai/)
- [GitHub Actions](https://docs.github.com/en/actions)
