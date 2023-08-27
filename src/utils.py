import numpy as np
from typing import Iterable, List, Collection, Dict
from pathlib import Path
from tqdm import tqdm
import os
import imageio
from PIL import Image

import proglog
proglog.default_bar_logger = lambda *args, **kwargs: proglog.MuteProgressBarLogger()

from moviepy.video.io.VideoFileClip import VideoFileClip, AudioFileClip

from .data_models import ClipData
from .models.image_to_text import ImageToTextModelService
from .models.transcriber import TranscriberModelService

TMP_AUDIO_PATH = 'tmp.mp3'


def split_video(video_path: Path, output_dir: Path, rtol: float=0.2):
    """Splits a video into continuous clips by relative difference between consecutive frames.
    
    The relative difference between two frames, `A`, `B` is calculated as
    `|A - B| / (|A| + |B|)`, which is always between 0 and 1 for real matrices.

    Args:
        video (th.Tensor): The video. Shape should be (frame_index, channel, height, width).
        rtol (float, optional): The maximum relative difference in a clip.
            When two consecutive frames have a relative difference larger than rtol,
            they are considered the end frame of one clip and the start frame of another.
            Defaults to 0.1.
    """
    
    assert rtol >= 0 and rtol <= 1, "rtol must be between 0 and 1!"
    
    def norm(x: np.ndarray):
        return np.sqrt((x ** 2).sum())

    def calc_frame_difference(frame1, frame2):
        return norm(frame1 - frame2) / (norm(frame1) + norm(frame2))

    video_reader = imageio.get_reader(video_path)
    metadata = video_reader.get_meta_data()
    video_clip = VideoFileClip(str(video_path))
    audio_clip = video_clip.audio
    fps, codec = video_clip.fps, metadata['codec']
    
    last_frame = next(iter(video_reader))

    clip_index = 0
    def get_tmp_path(number):
        return Path(output_dir / f"{number}_tmp.mp4")
    
    def get_output_path(number):
        return Path(output_dir / f"{number}.mp4")
    
    if not output_dir.is_dir():
        output_dir.mkdir()
        
    current_writer = imageio.get_writer(get_tmp_path(clip_index + 1), fps=fps, codec=codec, macro_block_size=1)

    def format_save_message(number, path):
        return f"Saved clip {number} to {path}."
    
    n_frames = metadata['duration'] * metadata['fps']

    start_time = 0.5 / fps
    
    progress = tqdm(video_reader, total=n_frames)
    for frame_idx, frame in enumerate(progress):
        progress.set_description(f'clip {clip_index + 1}')

        diff = calc_frame_difference(last_frame, frame)
        
        assert 0 <= diff < 1
        
        if diff > rtol:
            end_time = (frame_idx + 0.5) / fps
            
            current_writer.close()
            current_clip = VideoFileClip(str(get_tmp_path(clip_index + 1)))
            current_clip.audio = audio_clip.subclip(start_time, end_time)
            current_clip.write_videofile(str(get_output_path(clip_index + 1)))
            os.remove(get_tmp_path(clip_index + 1))
            
            clip_index += 1
            
            current_writer = imageio.get_writer(get_tmp_path(clip_index + 1), fps=fps, codec=codec, macro_block_size=1)
            current_writer.append_data(frame)
            
            start_time = end_time
            
        else:
            current_writer.append_data(frame)
        
        last_frame = frame
    
    end_time = (frame_idx + 0.5) / fps
    
    current_writer.close()
    current_clip = VideoFileClip(str(get_tmp_path(clip_index + 1)))
    current_clip.audio = audio_clip.subclip(start_time, end_time)
    current_clip.write_videofile(str(get_output_path(clip_index + 1)))
    os.remove(get_tmp_path(clip_index + 1))

    video_reader.close()

def get_arbitrary_image(video_path: Path):
    """Gets an arbitrary image from a video clip.
    
    Current implementation: get the first frame.

    Args:
        video_path (Path): _description_
    """
    
    with imageio.get_reader(video_path) as reader:
        frame = next(iter(reader))

    return Image.fromarray(frame)

def generate_clip_data(clip_paths: Collection[Path],
                       transcriber_class: type,
                       image_describer_class: type) -> Dict[Path, ClipData]:
    """Generate ASR audio transcriptions and screenshot descriptions for a set of video clips.

    Args:
        clip_paths (Collection[Path]): The paths to the video clips.
        transcriber (type): The class of the model used for audio transcription. MUST be subclass of TranscriberModelService.
        image_describer (type): The class of the model used for image captioning. MUST be subclass of ImageToTextModelService.

    Returns:
        Dict[Path, ClipData]: <video_clip_path, clip_data> pairs.
    """
    
    assert issubclass(transcriber_class, TranscriberModelService)
    assert issubclass(image_describer_class, ImageToTextModelService)
    
    # transcription
    transcriber: TranscriberModelService = transcriber_class()
    transcriptions = {}
    for clip_path in clip_paths:
        VideoFileClip(str(clip_path)).audio.write_audiofile(str(TMP_AUDIO_PATH))
        transcriptions[clip_path] = transcriber(TMP_AUDIO_PATH)
    
    # delete transcriber to free VRAM
    del transcriber

    # image captioning
    captioner: ImageToTextModelService = image_describer_class()
    captions = {}
    for clip_path in clip_paths:
        image = get_arbitrary_image(clip_path)
        captions[clip_path] = captioner(image)
    
    # delete image captioner to free VRAM
    del captioner
    
    return {
        ClipData(
            video_path=clip_path,
            audio_transcription_raw=transcriptions[clip_path],
            screenshot_description=captions[clip_path]
        ) for clip_path in clip_paths
    }
    