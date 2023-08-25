import numpy as np
from typing import Iterable
from pathlib import Path
from tqdm import tqdm
import imageio
import proglog

proglog.default_bar_logger = lambda *args, **kwargs: proglog.MuteProgressBarLogger()

from moviepy.video.io.VideoFileClip import VideoFileClip


def split_video(video_path: Path, output_dir: Path, rtol: float=0.2):
    """Splits a video into continuous clips by relative difference between consecutive frames.
    
    The relative difference between two frames, `A`, `B` is calculated as
    `|A - B| / (|A| + |B|)`, which is always between 0 and 1 for real matrices.
    `|A|` denotes Frobenius norm, i.e., $\sqrt{\sum_{i, j, k} A_{i, j, k}^2}$.

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

    # inter-frame inconsistency formula implementation
    def calc_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
        return norm(frame1 - frame2) / (norm(frame1) + norm(frame2))

    # load the video and get the metadata
    # use imageio to read each frame as a numpy array
    video_reader = imageio.get_reader(video_path)
    metadata = video_reader.get_meta_data()
    
    # use moviepy to create subclips to be written
    video_clip = VideoFileClip(str(video_path))
    
    fps, codec = video_clip.fps, metadata['codec']

    def get_output_path(number: int) -> Path:
        return Path(output_dir / f"{number}.mp4")
    
    # initiate loop-reused variables
    last_frame = next(iter(video_reader))
    start_time = 1 / fps
    clip_index = 0
    
    if not output_dir.is_dir():
        output_dir.mkdir()
        
    n_frames = metadata['duration'] * metadata['fps']
    
    progress = tqdm(video_reader, total=n_frames)
    for frame_idx, frame in enumerate(progress):
        progress.set_description(f'clip {clip_index + 1}')
        
        diff = calc_frame_difference(last_frame, frame)
        
        assert 0 <= diff < 1
        
        if diff > rtol:
            # two frames are too inconsistent -> split
            end_time = (frame_idx + 1) / fps
            
            sub_clip: VideoFileClip = video_clip.subclip(start_time, end_time)
            sub_clip.write_videofile(str(get_output_path(clip_index + 1)))

            clip_index += 1

            start_time = (frame_idx + 1) / fps
        
        last_frame = frame
    
    current_path = get_output_path(clip_index + 1)

    video_reader.close()
