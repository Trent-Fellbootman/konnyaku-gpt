import numpy as np
from typing import Iterable
from pathlib import Path
from tqdm import tqdm
import imageio


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
    fps, codec = metadata['fps'], metadata['codec']

    print(metadata.keys())

    last_frame = next(iter(video_reader))

    clip_index = 0
    def get_output_path(number):
        return Path(output_dir / f"{number}.mp4")
    
    if not output_dir.is_dir():
        output_dir.mkdir()
        
    current_writer = imageio.get_writer(get_output_path(clip_index + 1), fps=fps, codec=codec, macro_block_size=1)

    def format_save_message(number, path):
        return f"Saved clip {number} to {path}."
    
    n_frames = metadata['duration'] * metadata['fps']
    
    progress = tqdm(video_reader, total=n_frames)
    for frame in progress:
        diff = calc_frame_difference(last_frame, frame)
        
        assert 0 <= diff < 1
        
        if diff > rtol:
            current_writer.close()

            current_path = get_output_path(clip_index + 1)
            
            clip_index += 1
            
            current_writer = imageio.get_writer(get_output_path(clip_index + 1), fps=fps, codec=codec, macro_block_size=1)
            current_writer.append_data(frame)
            
            progress.set_description(f'clip {clip_index + 1}')
        else:
            current_writer.append_data(frame)
        
        last_frame = frame
    
    current_writer.close()
    current_path = get_output_path(clip_index + 1)

    video_reader.close()
