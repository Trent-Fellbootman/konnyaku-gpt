from typing import Sequence, Tuple
import openai

from .chat_llm import ChatCompletionService

import logging


class GPT35Turbo(ChatCompletionService):

    def __init__(self, max_retry_count: int) -> None:
        """Constructor.

        Args:
            max_retry_count (int): The maximum number to retry on each call.
        """
        
        super().__init__()

        self._max_retry_count = max_retry_count
    
    # override
    def call(self, messages: Sequence[Tuple[str, bool]]) -> str:
        # TODO: add error handling
        result = ''
        
        for _ in range(self._max_retry_count):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k",
                    messages=[
                        {"role": "user" if is_user else "assistant", "content": message}
                        for message, is_user in messages
                    ]
                )
                
                result = response['choices'][0]['message']['content']
                logging.info(f'GPT response: {result}')
                
                break
                
            except Exception:
                continue
        
        
        return result
    
    # override
    @staticmethod
    def get_description() -> str:
        return \
        """Cloud-deployed gpt-3.5-turbo.
        """
