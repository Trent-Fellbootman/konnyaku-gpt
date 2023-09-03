"""Subtitle generator that feeds multi-media information to language models to infer plausible subtitles."""

import json
from pathlib import Path
from typing import Sequence, Dict, List, Tuple, Any, Callable
import logging

from .subtitle_generator import SubtitleGenerator
from ..models import TranscriberModelService, ImageToTextModelService
from ..many_clips_transcription_correction import ManyClipsTranscriptionCorrector
from ..utils import compile_video_for_llm
from ..data_models import ClipData, ClipSetMetadata
from ..srt_export import export_to_srt


class MultiMediaLlmSubtitleGenerator(SubtitleGenerator):
    
    def __init__(self,
                 audio_transcriber_instantiator: Callable[[], TranscriberModelService],
                 frame_describer_instantiator: Callable[[], ImageToTextModelService],
                 transcription_corrector_instantiator: Callable[[], ManyClipsTranscriptionCorrector]):
        """Constructor.

        Args:
            audio_transcriber_instantiator (Callable[[], TranscriberModelService]): Audio transcriber instantiator.
                The return value of this instantiator is used as the ASR model for audio transcription.
            frame_describer_instantiator (Callable[[], ImageToTextModelService]): Frame describer instantiator.
                The return value of this instantiator is used as the model for image captioning.
            corrector (TranscriptionCorrector): Transcription corrector instantiator.
                The return value of this instantiator is used as the corrector for correcting the transcriptions from multi-media information input.
        """
        
        super().__init__()
        
        self._audio_transcriber_instantiator = audio_transcriber_instantiator
        self._frame_describer_instantiator = frame_describer_instantiator
        self._transcription_corrector_instantiator = transcription_corrector_instantiator
    
    # override
    def generate_subtitles(self, video_path: Path, output_path: Path, video_background: str, target_language: str | None=None,
                           workspace_path: Path | None=None, split_clip_rtol: float=0.4, save_every: int=10,
                           corrector_extra_arguments: Dict[str, Any]={}):
        """Generates subtitles for a video.

        Args:
            video_path (Path): The path to the video to generate subtitles from.
            output_path (Path): The subtitle output file path.
            video_background (str): The background information of the video.
            target_language (str | None, optional): The language that the subtitles should be in.
                "None" means "keep original language" (i.e., no translation). Defaults to None.
            workspace_path (Path | None, optional): The workspace path. Defaults to None.
                If None, then the workspace is created in the directory that contains the video,
                with the folder name being <video-base-name>_workspace.
            split_clip_rtol (float, optional): The relative tolerance value used when splitting the video into clips.
                Must be between 0 and 1.
            save_every (int, optional): When compiling the transcriptions & frame descriptions, automatic saving will occur every `save_every` clips.
            corrector_extra_arguments (Dict[str, Any], optional): Extra named arguments to pass to the corrector.
        """
        
        assert 0 <= split_clip_rtol <= 1, f'split_clip_rtol must be between 0 and 1, but got {split_clip_rtol}!'
        assert isinstance(save_every, int) and save_every >= 1, f'save_every must be a positive integer, but got {save_every}!'
        
        if workspace_path is None:
            workspace_path = video_path.absolute().resolve().parent / f'{video_path.stem}_workspace'

        if not workspace_path.exists():
            workspace_path.mkdir()
        
        # split video and generate audio transcriptions & frame descriptions
        logging.info('Splitting video and generating audio transcriptions & frame descriptions...')
        multimedia_info_compilation_workspace_path = workspace_path / 'multimedia_info'
        compile_video_for_llm(video_path, multimedia_info_compilation_workspace_path, self._audio_transcriber_instantiator, self._frame_describer_instantiator, split_clip_rtol, save_every)

        # assemble multimedia information
        with open(multimedia_info_compilation_workspace_path / 'clips/metadata.json', 'r') as f:
            clips_metadata: ClipSetMetadata = ClipSetMetadata.from_json(f.read())
        
        durations = [item.duration for item in clips_metadata.clips_metadata]

        with open(multimedia_info_compilation_workspace_path / 'transcriptions.json', 'r') as f:
            transcriptions = json.load(f)
        
        with open(multimedia_info_compilation_workspace_path / 'captions.json', 'r') as f:
            captions = json.load(f)
        
        assert isinstance(durations, List) and all(isinstance(d, float) for d in durations), 'Error: malformed durations data'
        assert isinstance(transcriptions, List) and all(isinstance(t, str) for t in transcriptions), 'Error: incorrect JSON structure in transcriptions data'
        assert isinstance(captions, List) and all(isinstance(c, str) for c in captions), 'Error: incorrect JSON structure in captions data'
        assert len(durations) == len(transcriptions) == len(captions), 'Error: lengths of durations, transcriptions, and captions are not the same'

        clips_data = [ClipData(duration, [transcription], caption) for duration, transcription, caption in zip(durations, transcriptions, captions)]

        # correct transcriptions
        # construct "additional information" from models' descriptions
        logging.info('Correcting transcriptions...')
        transcriber_description = self._audio_transcriber_instantiator().get_description()
        frame_describer_description = self._frame_describer_instantiator().get_description()
        
        corrector = self._transcription_corrector_instantiator()
        corrected_transcriptions = corrector.correct_transcriptions(
            clips_data, video_background,
            auxiliary_information=\
f"""The speech recognition model and image captioning models used to create transcriptions and frame descriptions may have special quirks that produce inaccurate outputs in particular ways.

For your reference, here is a description of the speech recognition model:

<speech-recognition-model-description-start>
{transcriber_description}
<speech-recognition-model-description-end>

Here is a description of the image captioning model:

<image-captioning-model-description-start>
{frame_describer_description}
<image-captioning-model-description-end>

Pay attention to the special behavior of the models; the models' quirks may result in inaccurate and even misleading outputs.
""",
            target_language=target_language,
            cache_path=workspace_path / 'transcription_correction_cache',
            **corrector_extra_arguments
        )
        
        # export subtitles
        logging.info('Exporting subtitles...')
        srt_string = export_to_srt(clips_metadata.clips_metadata, corrected_transcriptions)

        output_path.touch()
        
        with open(output_path, 'w') as f:
            f.write(srt_string)
        
        logging.info('Subtitle generation complete.')
