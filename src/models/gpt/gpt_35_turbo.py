from typing import Sequence, Tuple, Dict

from ..chat_completion import ChatCompletionService
from .openai_gpt_server import OpenAiGptServer


class GPT35Turbo(ChatCompletionService):

    def __init__(self, openai_gpt_server: OpenAiGptServer, context_length: str='4k', max_retry_count: int=3) -> None:
        """Constructor.

        Args:
            max_retry_count (int): The maximum number to retry on each call.
            context_length (str): The maximum context length. Either "4k" or "16k".
        """
        
        super().__init__()

        self._openai_server = openai_gpt_server
        self._max_retry_count = max_retry_count
        self._context_length = context_length
    
    # override
    def call(self, messages: Sequence[Tuple[str, bool]]) -> str:
        
        for _ in range(self._max_retry_count):
            try:
                result = self._openai_server.invoke('gpt-3.5-turbo' + ('-16k' if self._context_length == '16k' else ''), messages)
                return result
                
            except Exception as e:
                continue
        
        raise Exception(f'Max retry count of ({self._max_retry_count}) reached!')
        
    # override
    @staticmethod
    def get_description() -> str:
        return \
        """gpt-3.5-turbo, a.k.a, ChatGPT.
        """
