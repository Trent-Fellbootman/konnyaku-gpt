from typing import Sequence, Tuple, Dict

from ..chat_completion import ChatCompletionService
from .openai_gpt_server import OpenAiGptServer


class GPT4(ChatCompletionService):

    def __init__(self, openai_gpt_server: OpenAiGptServer, context_length: str='8k', max_retry_count: int=3) -> None:
        """Constructor.

        Args:
            max_retry_count (int): The maximum number to retry on each call.
            context_length (str): The maximum context length. Either "8k" or "32k".
        """
        
        super().__init__()

        self._openai_server = openai_gpt_server
        self._max_retry_count = max_retry_count
        self._context_length = context_length
    
    # override
    def call(self, messages: Sequence[Tuple[str, bool]]) -> str:
        for _ in range(self._max_retry_count):
            try:
                result = self._openai_server.invoke('gpt-4' + ('-32k' if self._context_length == '32k' else ''), messages)
                return result
                
            except Exception:
                continue
        
        raise Exception(f'Max retry count of ({self._max_retry_count}) reached!')
        
    # override
    @staticmethod
    def get_description() -> str:
        return \
        """GPT-4.
        """
