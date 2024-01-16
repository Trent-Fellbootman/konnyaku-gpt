from typing import Sequence
from ..data_models import ClipData
from .transcription_corrector import TranscriptionCorrector
import time

from numpy import random


class DummyCorrector(TranscriptionCorrector):
    
    def __init__(self, delay: float=0.5):
        super().__init__()

        self.delay = delay
    
    # override
    def correct_transcriptions(self, clips_data: Sequence[ClipData], video_background: str, auxiliary_information: str, target_language: str | None = None) -> Sequence[str]:
        time.sleep(self.delay)
        if random.random() < 0.5:
            return [f'dummy {i}: ' + "\n".join(clip_data.audio_transcriptions_raw) for i, clip_data in enumerate(clips_data)]
        else:
            raise Exception('Random error')
