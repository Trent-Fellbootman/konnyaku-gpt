from abc import ABC, abstractmethod
from ..data_models import ClipData
from typing import Sequence


class TranscriptionCorrector(ABC):
    
    @abstractmethod
    def correct_transcriptions(self, clips_data: Sequence[ClipData], video_background: str, auxiliary_information: str, target_language: str | None=None) -> Sequence[str]:
        """Correct the transcriptions.

        Args:
            clips_data (Sequence[ClipData]): The clips whose translations are to be corrected.
            video_background (str): The background information of the video that the clips come from.   
            auxiliary_information (str): Any auxiliary information like ASR & image-to-text model quirks.
            target_language (str | None): None if transcriptions should not be translated, the target language of translation otherwise.

        Returns:
            Sequence[str]: The corrected (and translated if should translate) transcriptions.
        """

        raise NotImplementedError()
