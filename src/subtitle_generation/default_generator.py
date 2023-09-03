"""Convenient class that wraps a multi-media LLM subtitle generator."""

import logging
from pathlib import Path
from .subtitle_generator import SubtitleGenerator
from .multimedia_llm_subtitle_generator import MultiMediaLlmSubtitleGenerator
from ..many_clips_transcription_correction import ManyClipsBatchedCorrector
from ..transcription_correction import SimpleCorrector, MultiRoundCorrector
from ..models.gpt import OpenAiGptServer, GPT35Turbo, GPT4
from ..models.whisper_cloud import WhisperCloud
from ..models.blip_large import BlipLarge


class DefaultGenerator(SubtitleGenerator):
    """A convenient class that wraps a multi-media LLM subtitle generator.
    """
    
    class QualityPresets:
        LOW = 'low'
        MEDIUM = 'medium'
        HIGH = 'high'
        VERY_HIGH = 'very-high'
    
    def __init__(self, quality_preset: str=QualityPresets.MEDIUM):
        """Constructs a default generator.

        Args:
            quality_preset (str): The quality of the generated subtitles.
                Either "low", "medium", "high", or "very-high".
                Defaults to "medium".
                "low" quality generator uses single-round correction & translation and GPT-3.5-Turbo in each transcription group.
                "medium" quality generator also uses single-round correction & translation but uses GPT-4 in transcription correction.
                "high" quality generator uses multi-round correction and employs GPT-4 to analyze the plot and suggest fixes to transcriptions.
                "very-high" quality generator uses multi-round correction and employs GPT-4 to analyze the plot, suggest fixes to transcriptions, and translate the transcriptions.
                The typical costs of "low", "medium", and "high" quality generators on a 20-minute video are $0.7, $4.5 and $15 (not tested), respectively.
        """
        
        self._gpt_server = OpenAiGptServer()

        # instantiate correctors based on quality preset
        match quality_preset:
            case self.QualityPresets.LOW:
                self._group_corrector = SimpleCorrector(
                    correction_backend=GPT35Turbo(self._gpt_server, context_length='16k'),
                    extraction_backend=GPT35Turbo(self._gpt_server, context_length='16k')
                )
            case self.QualityPresets.MEDIUM:
                self._group_corrector = SimpleCorrector(
                    correction_backend=GPT4(self._gpt_server, context_length='8k'),
                    extraction_backend=GPT35Turbo(self._gpt_server, context_length='16k')
                )
            case self.QualityPresets.HIGH:
                self._group_corrector = MultiRoundCorrector(
                    plot_analysis_backend=GPT4(self._gpt_server, context_length='8k'),
                    fix_suggestion_backend=GPT4(self._gpt_server, context_length='8k'),
                    fix_application_backend=GPT35Turbo(self._gpt_server, context_length='16k'),
                    translation_backend=GPT35Turbo(self._gpt_server, context_length='16k'),
                    transcription_extraction_backend=GPT35Turbo(self._gpt_server, context_length='16k'),
                )
            case self.QualityPresets.VERY_HIGH:
                self._group_corrector = MultiRoundCorrector(
                    plot_analysis_backend=GPT4(self._gpt_server, context_length='8k'),
                    fix_suggestion_backend=GPT4(self._gpt_server, context_length='8k'),
                    fix_application_backend=GPT35Turbo(self._gpt_server, context_length='16k'),
                    translation_backend=GPT4(self._gpt_server, context_length='32k'),
                    transcription_extraction_backend=GPT35Turbo(self._gpt_server, context_length='16k'),
                )
            case _:
                raise Exception(f'Unknown quality preset:{quality_preset} ')
        
        self._many_clips_transcription_corrector = ManyClipsBatchedCorrector(self._group_corrector)
        
        self._subtitle_generator = MultiMediaLlmSubtitleGenerator(
            audio_transcriber_instantiator=lambda: WhisperCloud(),
            frame_describer_instantiator=lambda: BlipLarge(),
            transcription_corrector_instantiator=lambda: self._many_clips_transcription_corrector
        )

    # override
    def generate_subtitles(self, video_path: Path, output_path: Path, video_background: str, target_language: str | None = None, workspace_path: Path | None = None):
        """Generates subtitles for a video.

        Args:
            video_path (Path): The path to the video to generate subtitles for.
            output_path (Path): The subtitle output file path.
            video_background (str): The background information of the video.
            target_language (str | None, optional): The language that the subtitles should be in. Defaults to None.
            workspace_path (Path | None, optional): The workspace path. Defaults to None.
                If None, then the workspace is created in the directory that contains the video,
                with the folder name being <video-base-name>_workspace.
        """
        
        start_money = self._gpt_server.money_spent["total"]
        current_money_spent = 0
        
        def on_group_complete():
            nonlocal current_money_spent
            new_total = self._gpt_server.money_spent["total"] - start_money
            logging.info(f'Cost on this group: ${new_total - current_money_spent: .2e}')
            current_money_spent = new_total
            logging.info(f'Total money spent: ${current_money_spent: .2e}')
        
        self._subtitle_generator.generate_subtitles(
            video_path=video_path, output_path=output_path, video_background=video_background, target_language=target_language,
            workspace_path=workspace_path,
            split_clip_rtol=0.4, save_every=10, corrector_extra_arguments={
                'min_target_clips_length': 40,
                'min_pre_context_length': 10,
                'min_post_context_length': 10,
                'group_completion_callback': on_group_complete
            }
        )
