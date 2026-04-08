import os
from openai import OpenAI
from dotenv import load_dotenv
from deepeval.models import DeepEvalBaseLLM

load_dotenv()

class CustomGroqModel(DeepEvalBaseLLM):
    def __init__(self, model: str):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("A variável de ambiente GROQ_API_KEY não foi encontrada.")
        
        self._model_name = model
        
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        
        response = client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"Groq: {self._model_name}"