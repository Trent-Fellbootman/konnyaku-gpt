import srt
import datetime
from typing import Sequence, List, Any, Dict, Tuple, Callable
import math


def simple_split_subtitles(subtitles: Sequence[srt.Subtitle], max_duration: datetime.timedelta) -> List[srt.Subtitle]:
    """Splits subtitles that are too long.
    
    The splitting scheme is simple; sentences are LIKELY to be broken into pieces.

    Args:
        subtitles (Sequence[srt.Subtitle]): Original subtitles.
        max_duration (float, optional): The maximum duration of each output subtitle, in seconds. Defaults to 10.
    """

    new_subtitles = []
    for subtitle in subtitles:
        remaining_text = subtitle.content
        current_start = subtitle.start
        splitted_subtitles = []

        while len(remaining_text) > 0:
            expected_text_length = math.floor(max_duration / (subtitle.end - current_start) * len(remaining_text))
            actual_text_length = min(expected_text_length, len(remaining_text))
            item_text = remaining_text[:actual_text_length]
            item_duration = len(item_text) / len(remaining_text) * (subtitle.end - current_start)

            splitted_subtitles.append(srt.Subtitle(
                index=len(new_subtitles) + len(splitted_subtitles),
                start=current_start,
                end=current_start + item_duration,
                content=item_text
            ))

            remaining_text = remaining_text[actual_text_length:]
            current_start += item_duration
        
        new_subtitles += splitted_subtitles
    
    return new_subtitles
