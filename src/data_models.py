from typing import Self, Dict, List
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod
import json


class JsonSerializable(ABC):
    @abstractmethod
    def to_json(self) -> str:
        raise NotImplementedError()
    
    @staticmethod
    @abstractmethod
    def from_json(self, json_data: str) -> Self:
        raise NotImplementedError()


@dataclass
class ClipData(JsonSerializable):
    video_path: str
    audio_transcription_raw: str
    screenshot_description: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=4)
    
    @staticmethod
    def from_json(json_data: str) -> Self:
        return ClipData(**json.loads(json_data))


@dataclass
class ClipsMetadata(JsonSerializable):
    """
    clip_paths (List[str]): The paths to the output clips. Indices correspond to clip ordering.
    """
    
    clip_paths: List[str]

    def to_json(self) -> str:
        return json.dumps(self.clip_paths, indent=4)
    
    @staticmethod
    def from_json(json_data: str) -> Self:
        return ClipsMetadata(json.loads(json_data))
