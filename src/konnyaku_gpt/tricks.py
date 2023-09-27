import srt
import datetime
from typing import Sequence, List, Any, Dict, Tuple, Callable
import math
from pathlib import Path


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

def simple_split_subtitle_file(input_srt: Path, output_srt: Path | None=None, max_duration: datetime.timedelta=datetime.timedelta(seconds=10)):
    """Split the subtitles. After splitting, the maximum duration of each subtitle will be max_duration.

    Args:
        input_srt (Path): The input SRT file path.
        output_srt (Path | None, optional): The output srt file path.
            `None` sets it to be the same as the input srt file. Defaults to None.
        max_duration (datetime.timedelta, optional): The maximum duration of each subtitle element in the output srt.
            Defaults to 10 seconds.
    """
    
    output_srt = output_srt if output_srt is not None else input_srt
    
    with open(input_srt, 'r') as f:
        subtitles = list(srt.parse(f.read()))

    splitted_subtitles = simple_split_subtitles(subtitles, max_duration=max_duration)

    output_srt.touch()
    
    with open(output_srt, 'w') as f:
        f.write(srt.compose(splitted_subtitles))
