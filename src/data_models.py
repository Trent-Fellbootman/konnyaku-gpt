from typing import Self, Dict
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class ClipData:
    video_path: Path
    audio_transcription_raw: str
    screenshot_description: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__)
    
    @staticmethod
    def from_json(json_data: str) -> Self:
        return ClipData(**json.loads(json_data))
