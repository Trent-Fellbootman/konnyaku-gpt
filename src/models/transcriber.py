"""Automatic speech recognition model service

Backend: whisper-small (local, cuda if available else cpu)
"""
from abc import ABC, abstractmethod
from pathlib import Path
from .base import ModelService



class TranscriberModelService(ModelService):
    """Abstraction for a model service which generates transcriptions from audio files.
    """
    
    @abstractmethod
    def call(self, audio_path: Path) -> str:
        """Generates a transcription of an audio file.
        
        Args:
            audio_path (Path): The path to the audio file.
        
        Returns:
            str: The transcription.
        """
        
        raise NotImplementedError()
