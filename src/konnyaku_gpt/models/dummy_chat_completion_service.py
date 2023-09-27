from typing import Sequence, Tuple
from .chat_completion import ChatCompletionService
from numpy import random
import time


class DummyChatCompletionService(ChatCompletionService):
    
    def __init__(self, success_prob: float=0.5, delay: float=1, return_message: str=''):
        super().__init__()

        self.success_prob = success_prob
        self.delay = delay
        self.return_message = return_message
        
    # override
    def call(self, messages: Sequence[Tuple[str, bool]]) -> str:
        time.sleep(self.delay)
        
        if random.random() < self.success_prob:
            return self.return_message
        else:
            raise Exception('Random error')
    
    # override
    @staticmethod
    def get_description() -> str:
        return """Dummy chat completion model that always returns the same message."""
