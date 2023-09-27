from abc import ABC, abstractmethod
from .base import ModelService
from PIL.Image import Image


class ImageToTextModelService(ModelService):
    """Abstraction for a model service which generates descriptions of images.
    """

    @abstractmethod
    def call(self, image: Image) -> str:
        """Generates a description of an image.

        Args:
            image (Image): The image.

        Returns:
            str: The description.
        """
        
        raise NotImplementedError()
