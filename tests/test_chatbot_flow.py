import os
import uuid
import pytest
import httpx
import deepeval
from dotenv import load_dotenv
# from tests.custom_metrics import CustomMetrics

load_dotenv()
try:
    api_key = os.getenv("CONFIDENT_API_KEY")
    if api_key and api_key != "your_actual_api_key_here":
        deepeval.login_with_confident_api_key(api_key)
        print("✅ Successfully logged into Confident AI")
    else:
        print("⚠️  CONFIDENT_API_KEY not set - results won't be uploaded")
        os.environ["DEEPEVAL_DISABLE_UPLOADS"] = "true"
except Exception as e:
    print(f"⚠️  Could not login to Confident AI: {e}")
    os.environ["DEEPEVAL_DISABLE_UPLOADS"] = "true"

from deepeval import assert_test, log_hyperparameters
from deepeval.dataset.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, BiasMetric, ToxicityMetric
from deepeval.test_case.llm_test_case import LLMTestCase

from groq_custom_model import CustomGroqModel
from operouter_custom_model import CustomOpenRouterModel
from gemini_custom_model import CustomGeminiModel

# --- Configuration ---
API_BASE_URL_CHAT = os.getenv("API_BASE_URL", "http://localhost:8000")
EVAL_MODEL_A = CustomGroqModel(model="llama-3.3-70b-versatile")
EVAL_MODEL_B = CustomOpenRouterModel(model="openai/gpt-4o")
# EVAL_MODEL_C = CustomGeminiModel(model="gemini-2.5-flash")

# --- Test Data (Goldens) ---
try:
    with open("data/context.txt", "r", encoding="utf-8") as f:
        JURIS_CONTEXT = f.read()
except FileNotFoundError:
    JURIS_CONTEXT = ""

GOLDENS = [
    Golden(
        input="Qual é o preço da consulta inicial?",
        expected_output="O preço da consulta inicial é R$ 850,00.",
        context=[JURIS_CONTEXT],
        retrieval_context=[JURIS_CONTEXT]
    ),
    Golden(
        input="Vocês trabalham com direito de família ou divórcio?",
        expected_output="Não possuo informações sobre este tópico.",
        context=[JURIS_CONTEXT],
        retrieval_context=[JURIS_CONTEXT]
    ),
]

# --- Test Client ---
class TestChatBotClient:
    """A client to interact with the Chatbot API for testing."""
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.Client(base_url=self.base_url)

    def get_response_for_golden(self, golden: Golden) -> str:
        """Sends a request to the chatbot and returns the response."""
        session_id = str(uuid.uuid4())
        headers = {"session-id": session_id}
        payload = {
            "question": golden.input,
            "history": [],
            "model": "llama-3.1-8b-instant"
        }
        try:
            response = self.client.post("/chat/", headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get("response", "")
        except httpx.RequestError as e:
            pytest.fail(f"API request failed: {e}")
        except Exception as e:
            pytest.fail(f"An unexpected error occurred in TestChatBotClient: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

dataset = EvaluationDataset()
test_cases = []
with TestChatBotClient(base_url=API_BASE_URL_CHAT) as bot:
    for golden in GOLDENS:
        actual_output = bot.get_response_for_golden(golden)
        test_case = LLMTestCase(
            name=golden.input,
            input=golden.input,
            actual_output=actual_output,
            expected_output=golden.expected_output,
            context=golden.context,
            retrieval_context=golden.retrieval_context
        )
        dataset.test_cases += [test_case]

@log_hyperparameters
def hyperparameters():
    SYSTEM_PROMPT_TEMPLATE = """Você é o "JurisBot", o assistente virtual da Juris Consultoria"""
    return {
        "model": "llama-3.3-70b-versatile",
        "prompt_template": SYSTEM_PROMPT_TEMPLATE,
        "temperature": 0.7
    }

@pytest.mark.parametrize("test_case", dataset)
def test_chatbot_singleturn(test_case: LLMTestCase):
    """
    Evaluates chatbot responses for single-turn conversations.
    """
    assert_test(
        test_case,
        [
            AnswerRelevancyMetric(model=EVAL_MODEL_A, threshold=0.25),
            BiasMetric(model=EVAL_MODEL_A, threshold=0.25),
            ToxicityMetric(model=EVAL_MODEL_A, threshold=0.25),
            FaithfulnessMetric(model=EVAL_MODEL_A, threshold=0.25),
            # CustomMetrics.JurisBotComprehensive(model=EVAL_MODEL_A, threshold=0.25),
            # CustomMetrics.JurisBotComprehensiveDAG(model=EVAL_MODEL_C, threshold=0.25)
        ]
    )