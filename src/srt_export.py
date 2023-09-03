from typing import Sequence, Tuple
import datetime
import srt

from .data_models import ClipMetaData


def export_to_srt(clips_metadata: Sequence[ClipMetaData], transcriptions: Sequence[str]) -> str:
    assert len(clips_metadata) == len(transcriptions), 'Length of clips_metadata and transcriptions must be the same!'

    subtitles = [srt.Subtitle(i + 1,
                              datetime.timedelta(seconds=clip_metadata.clip_range[0]),
                              datetime.timedelta(seconds=clip_metadata.clip_range[1]),
                              transcription)
                 for i, (clip_metadata, transcription) in enumerate(zip(clips_metadata, transcriptions))]
    
    return str(srt.compose(subtitles))
