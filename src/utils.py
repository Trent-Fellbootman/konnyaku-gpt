import logging
import numpy as np
from typing import Iterable, List, Collection, Dict, Any, Callable, Tuple
from pathlib import Path
from tqdm import tqdm
import os
import imageio
from PIL import Image
import json
import shutil

import proglog
proglog.default_bar_logger = lambda *args, **kwargs: proglog.MuteProgressBarLogger()

from moviepy.video.io.VideoFileClip import VideoFileClip, AudioFileClip

from .data_models import ClipMetaData, ClipSetMetadata
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
    
    clips_metadata: List[ClipMetaData] = []
    
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
            clips_metadata.append(ClipMetaData(
                index=clip_index,
                path=get_output_path(clip_index + 1).name,
                clip_range=(start_time, end_time)
            ))
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
    clips_metadata.append(ClipMetaData(
        index=clip_index,
        path=get_output_path(clip_index + 1).name,
        clip_range=(start_time, end_time)
    ))
    os.remove(get_tmp_path(clip_index + 1))

    video_reader.close()
    
    with open(output_dir / 'metadata.json', 'x') as f:
        f.write(ClipSetMetadata(clips_metadata).to_json())

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

def transcribe_clips(clips_dir: Path, transcriber: TranscriberModelService, output_filepath: Path, save_every: int=10):
    try:
        output_filepath.touch()
        with open(output_filepath, 'r') as f:
            transcriptions = json.loads(f.read())
            
        assert isinstance(transcriptions, List) and all(isinstance(t, str) for t in transcriptions)
    except Exception:
        output_filepath.touch()
        transcriptions = []
    
    with open(clips_dir / 'metadata.json', 'r') as f:
        clip_paths: List[str] = [str(item.path) for item in ClipSetMetadata.from_json(f.read()).clips_metadata]
    
    progress = tqdm(list(enumerate(clip_paths)))
    
    def flush():
        with open(output_filepath, 'w') as f:
            f.write(json.dumps(transcriptions, indent=4, ensure_ascii=False))

    for i, clip_path in progress:
        if i < len(transcriptions):
            continue
        
        with VideoFileClip(str(clips_dir / clip_path)) as clip:
            clip.audio.write_audiofile(str(TMP_AUDIO_PATH))
            
        text = transcriber(TMP_AUDIO_PATH)
        transcriptions.append(text)
        progress.set_description(f'clip {i + 1}/{len(clip_paths)}: {text}')
        
        if (i + 1) % save_every == 0:
            flush()
    
    flush()

def describe_clips_screenshots(clips_dir: Path, captioner: ImageToTextModelService, output_filepath: Path, save_every: int=10):
    try:
        output_filepath.touch()
        with open(output_filepath, 'r') as f:
            captions = json.loads(f.read())
            
        assert isinstance(captions, List) and all(isinstance(t, str) for t in captions)
    except Exception:
        captions = []
    
    with open(clips_dir / 'metadata.json', 'r') as f:
        clip_paths: List[str] = [str(item.path) for item in ClipSetMetadata.from_json(f.read()).clips_metadata]
    
    progress = tqdm(list(enumerate(clip_paths)))
    
    def flush():
        with open(output_filepath, 'w') as f:
            f.write(json.dumps(captions, indent=4))

    for i, clip_path in progress:
        if i < len(captions):
            continue
        
        image = get_arbitrary_image(clips_dir / clip_path)
        caption = captioner(image)
        captions.append(caption)
        progress.set_description(f'clip {i + 1}/{len(clip_paths)}: {caption}')
        
        if (i + 1) % save_every == 0:
            flush()
    
    flush()

def parse_video_clips(clips_dir: Path,
                      transcriber_instantiator: Callable[[], TranscriberModelService],
                      image_describer_instantiator: Callable[[], ImageToTextModelService],
                      transcriptions_file: Path,
                      screenshot_descriptions_file: Path,
                      save_every: int=10):
    # transcribe clips
    logging.info('Transcribing clips...')
    transcriber = transcriber_instantiator()
    transcribe_clips(clips_dir, transcriber, transcriptions_file, save_every)

    # create screenshot descriptions for clips
    logging.info('Creating screenshot descriptions for clips...')
    captioner = image_describer_instantiator()
    describe_clips_screenshots(clips_dir, captioner, screenshot_descriptions_file, save_every)

def compile_video_for_llm(video_path: Path,
                          output_dir: Path,
                          transcriber_instantiator: Callable[[], TranscriberModelService],
                          image_describer_instantiator: Callable[[], ImageToTextModelService],
                          rtol: float=0.4,
                          save_every: int=10) -> None:
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
            
        transcriptions.json: the transcription of the clips
        captions.json: the captions of arbitrary screenshots fromthe clips

    Args:
        video_path (Path): Path to the video.
        output_dir (Path): Output directory root path.
        transcriber_instantiator (Callable[[], TranscriberModelService]): The return value is used as the ASR model for audio transcription.
        image_describer_instantiator (Callable[[], ImageToTextModelService]): The return value is used as the model for image captioning.
        rtol (float): `rtol` for image splitting.
    """
    
    if not output_dir.exists():
        output_dir.mkdir()
    
    clips_dir = output_dir / 'clips'
    
    if not (clips_dir.exists() and (clips_dir / 'metadata.json').exists()):
        if clips_dir.exists():
            shutil.rmtree(clips_dir)

        # split video into clips
        logging.info(f'Splitting video: {video_path}')
        split_video(video_path, clips_dir, rtol=rtol)
        logging.info(f'Video clips saved to {clips_dir}.')

    # ASR & image captioning
    parse_video_clips(clips_dir, transcriber_instantiator, image_describer_instantiator,
                      output_dir / 'transcriptions.json', output_dir / 'captions.json',
                      save_every=save_every)
