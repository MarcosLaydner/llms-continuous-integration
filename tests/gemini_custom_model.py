import os
import google.generativeai as genai
from deepeval.models.base_model import DeepEvalBaseLLM
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class CustomGeminiModel(DeepEvalBaseLLM):
    """
    CustomGeminiModel is a DeepEval-compatible model that uses the Google Gemini API.
    """
    def __init__(self, model: str):
        """
        Initializes the CustomGeminiModel.

        Args:
            model (str): The name of the Gemini model to use (e.g., "gemini-1.5-flash").

        Raises:
            ValueError: If the GOOGLE_API_KEY environment variable is not set.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("The GOOGLE_API_KEY environment variable was not found.")

        self._model_name = model
        
        # Configure the Gemini client
        genai.configure(api_key=api_key)
        
        self.client = genai.GenerativeModel(self._model_name)
        super().__init__(model=self._model_name)

    def load_model(self, *args, **kwargs):
        """
        Returns the Gemini client instance.
        """
        return self.client

    def generate(self, prompt: str) -> str:
        """
        Generates a response from the Gemini LLM.
        """
        try:
            response = self.client.generate_content(prompt)
            return response.text
        except Exception as e:
            # Handle potential API errors
            error_message = (
                f"{self.get_model_name()}.generate - Error during Gemini API call. "
                f"Error: {type(e).__name__}: {e}."
            )
            raise RuntimeError(error_message) from e

    async def a_generate(self, prompt: str) -> str:
        """
        Asynchronously generates a response from the Gemini LLM.
        """
        try:
            response = await self.client.generate_content_async(prompt)
            return response.text
        except Exception as e:
            # Handle potential API errors
            error_message = (
                f"{self.get_model_name()}.a_generate - Error during Gemini API call. "
                f"Error: {type(e).__name__}: {e}."
            )
            raise RuntimeError(error_message) from e

    def get_model_name(self) -> str:
        """
        Returns a display name for the model.
        """
        return f"Gemini: {self._model_name}"