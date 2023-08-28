import numpy as np
from typing import Iterable, List, Collection, Dict, Any, Callable
from pathlib import Path
from tqdm import tqdm
import os
import imageio
from PIL import Image
import json

import proglog
proglog.default_bar_logger = lambda *args, **kwargs: proglog.MuteProgressBarLogger()

from moviepy.video.io.VideoFileClip import VideoFileClip, AudioFileClip

from .data_models import ClipData, ClipsMetadata
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
    
    assert not output_dir.exists(), "Output directory already exists!"
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
    
    clip_paths: List[Path] = []
    
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
            clip_paths.append(get_output_path(clip_index + 1))
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
    clip_paths.append(get_output_path(clip_index + 1))
    os.remove(get_tmp_path(clip_index + 1))

    video_reader.close()
    
    with open(output_dir / 'metadata.json', 'x') as f:
        f.write(ClipsMetadata([str(clip_path.name) for clip_path in clip_paths]).to_json())

def get_arbitrary_image(video_path: Path) -> Image.Image:
    """Gets an arbitrary image from a video clip.
    
    Current implementation: get the first frame.

    Args:
        video_path (Path): The path to the video.
    
    Returns:
        Image.Image: The image.
    """
    
    with imageio.get_reader(video_path) as reader:
        frame = next(iter(reader))

    return Image.fromarray(frame)

def generate_clips_data(clip_paths: Collection[Path],
                       transcriber_instantiator: Any,
                       image_describer_instantiator: Any) -> Dict[Path, ClipData]:
    """Generate ASR audio transcriptions and screenshot descriptions for a set of video clips.

    Args:
        clip_paths (Collection[Path]): The paths to the video clips.
        transcriber_instantiator (Any): The return value of calling this function with no arguments is used as the model for audio transcription.
            The return value MUST be an instance of TranscriberModelService.
        image_describer_instantiator (Any): The return value of calling this function with no arguments is used as the model for image captioning.
            The return value MUST be an instance of ImageToTextModelService.

    Returns:
        Dict[Path, ClipData]: <video_clip_path, clip_data> pairs.
    """
    
    # transcription & durations
    transcriber: TranscriberModelService = transcriber_instantiator()
    transcriptions = {}
    durations = {}
    
    for clip_path in tqdm(clip_paths, desc="Generating audio transcriptions..."):
        clip = VideoFileClip(str(clip_path))
        clip.audio.write_audiofile(str(TMP_AUDIO_PATH))
        transcriptions[clip_path] = list(transcriber(TMP_AUDIO_PATH))
        durations[clip_path] = clip.duration
    
    # delete transcriber to free VRAM
    del transcriber

    # image captioning
    captioner: ImageToTextModelService = image_describer_instantiator()
    captions = {}
    for clip_path in tqdm(clip_paths, desc="Generating screenshot descriptions..."):
        image = get_arbitrary_image(clip_path)
        captions[clip_path] = captioner(image)
    
    # delete image captioner to free VRAM
    del captioner
    
    return {
        clip_path: ClipData(
            duration=durations[clip_path],
            audio_transcriptions_raw=transcriptions[clip_path],
            screenshot_description=captions[clip_path]
        ) for clip_path in clip_paths
    }

def compile_video_for_llm(video_path: Path,
                          output_dir: Path,
                          transcriber_instantiator: Callable[[], TranscriberModelService],
                          image_describer_instantiator: Callable[[], ImageToTextModelService],
                          rtol: float=0.4) -> None:
    """Compiles LLM-feedable data from a video. Steps include:
    
    1. Split the video into clips;
    2. For each clip, generates an audio transcription and a description of an arbitrarily picked frame from the clip.
    
    The output directory hierarchy is as follows:
    
    output-root/
        clips/
            metadata.json - metadata of the clips
            1.mp4 - clip 1
            2.mp4 - clip 2
            ...
            
        clips_data.json - duration, audio transcription and screenshot description for each clip

    Args:
        video_path (Path): Path to the video.
        output_dir (Path): Output directory root path.
        transcriber_instantiator (Callable[[], TranscriberModelService]): The return value is used as the ASR model for audio transcription.
        image_describer_instantiator (Callable[[], ImageToTextModelService]): The return value is used as the model for image captioning.
        rtol (float): `rtol` for image splitting.
    """
    
    assert not output_dir.exists(), "Output directory already exists!"
    
    output_dir.mkdir()
    clips_dir = output_dir / 'clips'
    
    # split video into clips
    print(f'Splitting video: {video_path}')
    split_video(video_path, clips_dir, rtol=rtol)
    print(f'Video clips saved to {clips_dir}.')

    # ASR & image captioning
    with open(clips_dir / 'metadata.json', 'r') as f:
        clip_paths = json.loads(f.read())
    
    clip_paths = [(clips_dir / clip_path).absolute().resolve() for clip_path in clip_paths]

    clips_data = generate_clips_data(clip_paths, transcriber_instantiator, image_describer_instantiator)

    # serialize clips data
    clips_data = [clips_data[clip_path].as_pytree() for clip_path in clip_paths]

    clips_data_path = output_dir / 'clips_data.json'
    
    with open(clips_data_path, 'x') as f:
        f.write(json.dumps(clips_data, indent=4))
    
    print(f'ASR outputs & screenshot descriptions saved as {clips_data_path.absolute().resolve()}')
