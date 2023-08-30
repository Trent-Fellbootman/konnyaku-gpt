from typing import Self, Dict, List, Tuple
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
    """`duration` is in seconds.
    """

    duration: float
    audio_transcriptions_raw: List[str]
    screenshot_description: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=4)
    
    @staticmethod
    def from_json(json_data: str) -> Self:
        return ClipData(**json.loads(json_data))
    
    def as_pytree(self) -> Dict[str, float | str]:
        return self.__dict__
    
    @staticmethod
    def from_pytree(data: Dict[str, float | str]) -> Self:
        return ClipData(**data)


@dataclass
class ClipMetaData(JsonSerializable):
    """`clip_range` is in seconds.
    
    `index` starts from 0.
    """

    index: int
    path: Path
    clip_range: Tuple[float, float]

    def as_pytree(self) -> str:
        return {
            'index': self.index,
            'path': str(self.path),
            'clip_range': list(self.clip_range)
        }
    
    @staticmethod
    def from_pytree(data: Dict) -> Self:
        return ClipMetaData(
            index=data['index'],
            path=Path(data['path']),
            clip_range=tuple(data['clip_range'])
        )
    
    def to_json(self) -> str:
        return json.dumps(self.as_pytree(), indent=4)
    
    @staticmethod
    def from_json(json_data: str) -> Self:
        return ClipMetaData.from_pytree(json.loads(json_data))
    
    @property
    def duration(self) -> float:
        return self.clip_range[1] - self.clip_range[0]


@dataclass
class ClipSetMetadata(JsonSerializable):
    clips_metadata: List[ClipMetaData]

    def to_json(self) -> str:
        return json.dumps([item.as_pytree() for item in self.clips_metadata], indent=4)
    
    @staticmethod
    def from_json(json_data: str) -> Self:
        return ClipSetMetadata([ClipMetaData.from_pytree(item) for item in json.loads(json_data)])
