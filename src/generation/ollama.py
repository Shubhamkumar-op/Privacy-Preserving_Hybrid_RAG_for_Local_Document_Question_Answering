from autogen import ConversableAgent

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from src.generation.prompts import build_prompt


class LocalGenerator:
    def __init__(self, model: str = OLLAMA_MODEL):
        self.agent = ConversableAgent(
            name="LocalRAGAssistant",
            system_message="Answer using only the supplied document context.",
            llm_config={
                "model": model,
                "base_url": OLLAMA_BASE_URL,
                "api_type": "ollama",
            },
        )

    def generate(self, question: str, contexts) -> str:
        prompt = build_prompt(question, contexts)
        response = self.agent.generate_reply(
            messages=[{"role": "user", "name": "User", "content": prompt}]
        )
        return response.get("content", "No answer generated.")
