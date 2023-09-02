from .base import ModelService
from abc import ABC, abstractmethod
from typing import Sequence, Tuple


class ChatCompletionService(ModelService):
    """Represents a model service for chat-completion.
    """

    @abstractmethod
    def call(self, messages: Sequence[Tuple[str, bool]]) -> str:
        """Uses the model service to infer the next chat message.

        Args:
            messages (Sequence[Tuple[str, bool]]): The chat message history.
                Each element is a tuple of a message and a boolean indicating its sender
                (True for the user, False for the bot).

        Returns:
            str: The next bot-sent message.
        """
        
        raise NotImplementedError()
