from abc import ABC, abstractmethod, abstractclassmethod


class ModelService(ABC):
    """Represents a callable deep learning model service.
    """

    @abstractmethod
    def call(self, *args, **kwargs):
        """Invokes the model service on some inputs.
        """
        
        raise NotImplementedError()
    
    @abstractmethod
    def get_description() -> str:
        """Returns a description of the model.

        This can include name, limitations, etc.
        
        ! Description MAY be multi-line.

        Returns:
            str: The description.
        """
        
        raise NotImplementedError()
    
    def __call__(self, *args, **kwargs):
        """Invokes the model service on some inputs.
        """
        
        return self.call(*args, **kwargs)
