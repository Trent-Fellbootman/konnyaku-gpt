from typing import Sequence, Tuple
from .chat_llm import ChatCompletionService


class GPT35Turbo(ChatCompletionService):
    
    # override
    def call(self, messages: Sequence[Tuple[str, bool]]) -> str:
        return super().call(messages)
    
    # override
    @staticmethod
    def get_description() -> str:
        return \
        """Cloud-deployed gpt-3.5-turbo.
        """
