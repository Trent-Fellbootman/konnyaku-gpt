from typing import Collection, Dict, List, Any, Sequence
from .data_models import ClipData

from .models.chat_llm import ChatCompletionService


class TranscriptionCorrector:
    """A transcription corrector which computes corrected transcription based on inaccurate transcriptions and screenshot descriptions.
    
    The intended use is to use one instance of this class for one video.
    """
    
    def __init__(self, video_background: str, asr_description: str, captioner_description: str, chat_backend: ChatCompletionService):
        """Constructor.

        Args:
            video_background (str): Background information about the video.
            asr_description (str): Information about the ASR model used to generate the initial transcriptions.
            captioner_description (str): Information about the image captioning model used to generate descriptions for screenshots of the clips.
            chat_backend (ChatCompletionService): A model service for chat-completion which will be used to compute corrected transcriptions.
        """

        self.video_background = video_background
        self.asr_description = asr_description
        self.captioner_description = captioner_description
        self.chat_backend = chat_backend
    
    def correct_transcriptions(self, context_pre: Sequence[ClipData], targets: Sequence[ClipData], context_post: Sequence[ClipData]) -> Sequence[str]:
        """Corrects the transcriptions of the clips in `targets`.

        Args:
            context_pre (Sequence[ClipData]): Data about the clips before `targets`.
            targets (Sequence[ClipData]): Data about the clips to be transcribed.
            context_post (Sequence[ClipData]): Data about the clips after `targets.

        Returns:
            Sequence[str]: The corrected transcriptions.
                Each element corresonponds to a clip in `targets`.
        """
        
        correction_prompt = self._build_transcription_correction_prompt(context_pre, targets, context_post)
        correction_output: str = self.chat_backend([(correction_prompt, True)])
        formatting_prompt = self._build_transcription_formatting_prompt(correction_output, len(targets))
        formatting_output: str = self.chat_backend([formatting_prompt])
        
        transcriptions = formatting_output.split('\n' * 2)

        return transcriptions

    def _build_transcription_correction_prompt(
        self,
        context_pre: Sequence[ClipData],
        targets: Sequence[ClipData],
        context_post: Sequence[ClipData]) -> str:
        """Builds the prompt used to be fed into an LLM and retrieve corrected transcriptions.
        
        The output of the LLM when fed with the returned value is expected to contain the corrected transcriptions,
        but has no specific machine-parsable format.

        Args:
            context_pre (Sequence[ClipData]): Data about the clips before `targets`.
            targets (Sequence[ClipData]): Data about the clips to be transcribed.
            context_post (Sequence[ClipData]): Data about the clips after `targets.

        Returns:
            str: The prompt ready to be fed into an LLM.
        """
        
        def clip_formatter(data: ClipData, index: int, is_target: bool) -> str:
            headline = f"Clip {index}{' (you need to correct the transcription for this clip)' if is_target else ''}:"
            asr_part = f"Inaccurate transcription from ASR:\n" + '\n'.join(data.audio_transcriptions_raw)
            screenshot_part = f"Inaccurate description of an arbitrarily picked frame within the clip:\n" + data.screenshot_description
            
            return '\n'.join([headline, asr_part, screenshot_part])
        
        clip_args = [
            (item, i, len(context_pre) <= i < len(context_pre) + len(targets))
            for i, item in enumerate(list(context_pre) + list(targets) + list(context_post))
        ]
        
        clips_data_part = ('\n' * 2).join(clip_formatter(*args) for args in clip_args)

        return \
f"""I am trying to create subtitles for a video.
To achive this goal, I split the video into many clips and
used an Automatic Speech Recognition (ASR) model to transcribe the audio of each clip.
However, due to the short length of each clip (which means the ASR model is provided almost no conntextual information on each call)
and the inferior performance of the ASR model, the generated transcriptions are VERY inaccurate.

Notice that the video may contain multiple speakers, and some of them may even speak simultaneously.
Also, the video is splitted with a very simple algorithm.
As a result, it is usual for one clip to contain multiple sentences spoken by multiple speakers,
or just an incomplete part of one sentence spoken by one speaker.
Hence, you should NOT assume that each clip always contains exactly one complete sentence.
If you think a clip only contains part of a sentence, you should also provide part of the corrected transcription accordingly.

To be more specific, here is a description of the ASR model:

{self.asr_description}

Therefore, I need you to use logical reasoning (and your imagination when information is insufficient) to infer what is going on in the video,
and then correct the transcriptions so that they look logical and make sense when viewed together.

I splitted the video into many clips and I don't expect you to provide a transcription for all of them all at once.
For now, you only need to correct the transcription for part of the clips.

To give you some context,
I will also provide you with the information of {len(context_pre)} clips before and {len(context_post)} clips after those you need to transcribe.

Furthermore, I selected an arbitrary frame from each clip and used an image-to-text model to create a description of that frame.
These descriptions will also be given to you so that you have access to "visual information" of the clips.
However, the image-to-text model is also VERY inaccurate, so DO NOT rely on it too much.

For your reference, here is a description of the image-to-text model:

{self.captioner_description}

Also, you may find the following background information of the video helpful:

{self.video_background}

Now, here are the information of the clips (0 indexed, indices correspond to the order in which the clips appears in the video):

{clips_data_part}

Please correct the transcriptions for clip {len(context_pre)}-{len(context_pre) + len(targets) - 1}.
Although there are likely to be multiple characters speaking, you DO NOT need to separate the speech of each speaker.
Just provide the audio transcriptions of entire clips.

Keep in mind that both the ASR model and the image-to-text model are VERY INACCURATE,
and their ONLY job is to convert the video information into text which you can process.
DO NOT rely on their outputs, as it is YOUR responsibility to understand the video.
Please use logical reasoning and your imagination to infer (or imagine) what is ACTUALLY going on in the clips,
and give me the audio transcriptions that seem most REASONABLE and PLAUSIBLE to you.

You should provide YOUR transcriptions for clip {len(context_pre)}-{len(context_pre) + len(targets) - 1} ONLY.
DO NOT include ASR outputs, image-to-text outputs, or information about contextual clips.
However, please do indicate which clip each of your transcription corresponds to.
"""


    def _build_transcription_formatting_prompt(self, transcription_output: str, n_clips: int) -> str:
        """Builds the prompt used to be fed into an LLM and retrieve corrected, formatted transcriptions.

        Args:
            transcription_output (str): LLM's output containing the transcriptions.
            n_clips (int): The (expected) number of clips included in `transcription_output`.

        Returns:
            str: The prompt ready to be fed into the LLM.
        """
        
        return \
f"""I used an LLM to correct transcriptions for a number of audio clips.
Please format the LLM output so that I can use some simple code to retrieve the transcriptions in a machine-friendly form.

The transcriptions in the LLM output are ordered by clip indices (not necessarily starting from 0 or 1),
and I want you to output the transcription CONTENT of each clip IN ORDER (lower indices first),
leaving a SINGLE blank lines between transcriptions of each two consecutive clips
(If a transcription of a certain clip already contains blank lines, remove them to avoid ambiguity).

For example, if the LLM output is:

    Sure! Here are my transcriptions for clip 15-17:
    
    Clip 15:
    blah1 blah1 blah1
    
    Clip 17:
    blah2
    
    blah2
    blah2
    
    Clip 16:
    blah3 blah3 blah3
    
Then your output should be (do not actually contain the indentations. They are for pretty-printing purposes only):

    blah1 blah1 blah1
    
    blah3 blah3 blah3
    
    blah2
    blah2
    blah2

Notice that "blah3" actually appears before "blah2", because "blah3" corresponds to clip 16, while "blah2" corresponds to clip 17.

Now, the LLM output I got is:

{transcription_output}

Please output the formatted transcription ONLY and NOTHING ELSE.
Also, keep the original transcriptions and do not try to correct them even if they seem grammatically incorrect.
"""
