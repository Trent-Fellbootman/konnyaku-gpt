from abc import ABC, abstractmethod


class ModelService(ABC):
    """Represents a callable deep learning model service.
    """

    @abstractmethod
    def call(self, *args, **kwargs):
        """Invokes the model service on some inputs.
        """
        
        raise NotImplementedError()
    
    def __call__(self, *args, **kwargs):
        """Invokes the model service on some inputs.
        """
        
        return self.call(*args, **kwargs)
