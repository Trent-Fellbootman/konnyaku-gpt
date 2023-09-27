from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Sequence, Set, Any, Tuple, Dict

from ..data_models import ClipData


class ManyClipsTranscriptionCorrector(ABC):
    """This type of correctors deals with large number of clips, like clips from a whole video.
    """
    
    @abstractmethod
    def correct_transcriptions(self, clips_data: Sequence[ClipData], video_background: str, auxiliary_information: str, target_language: str | None, cache_path: Path | None, *args, **kwargs) -> Sequence[str]:
        """Correct the transcriptions.

        Args:
            clips_data (Sequence[ClipData]): The clips whose translations are to be corrected.
            video_background (str): The background information of the video that the clips come from.   
            auxiliary_information (str): Any auxiliary information like ASR & image-to-text model quirks.
            target_language (str | None): None if transcriptions should not be translated, the target language of translation otherwise.
            cache_path (Path | None): The path to cache intermediate results.
                Implementors of this base class may disregard this parameter but must include it in the parameter list.

        Returns:
            Sequence[str]: The corrected (and translated if should translate) transcriptions.
        """
        
        raise NotImplementedError()
