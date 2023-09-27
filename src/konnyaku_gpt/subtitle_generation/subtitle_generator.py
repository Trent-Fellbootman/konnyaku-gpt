"""Subtitle generator base class."""

from abc import ABC, abstractmethod
from pathlib import Path

class SubtitleGenerator(ABC):
    
    @abstractmethod
    def generate_subtitles(self, video_path: Path, output_path: Path, video_background: str, target_language: str | None=None, *args, **kwargs):
        """Generates subtitles for a video.

        Args:
            video_path (Path): The path to the video to generate subtitles from.
            output_path (Path): The subtitle output file path.
            video_background (str): The background information of the video.
            target_language (str | None, optional): The language that the subtitles should be in.
                "None" means "keep original language" (i.e., no translation). Defaults to None.
        """
        
        raise NotImplementedError()