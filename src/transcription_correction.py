import sys
from typing import Collection, Dict, List, Any, Sequence
import json

from .data_models import ClipData
from .models.chat_llm import ChatCompletionService


class TranscriptionCorrector:
    """A transcription corrector which computes corrected transcription based on inaccurate transcriptions and screenshot descriptions.
    
    The intended use is to use one instance of this class for one video.
    """
    
    def __init__(self, asr_description: str, captioner_description: str, chat_backend: ChatCompletionService,
                 max_retry_count: int=3):
        """Constructor.

        Args:
            asr_description (str): Information about the ASR model used to generate the initial transcriptions.
            captioner_description (str): Information about the image captioning model used to generate descriptions for screenshots of the clips.
            chat_backend (ChatCompletionService): A model service for chat-completion which will be used to compute corrected transcriptions.
            max_retry_count (int): The maximum number to retry on each group of clips. If reached, the clip group will be skipped and no transcription
                will be generated for this group.
        """

        self.asr_description = asr_description
        self.captioner_description = captioner_description
        self.chat_backend = chat_backend
        self.max_retry_count = max_retry_count
   
    def correct_transcriptions(self,
                               clips_data: Sequence[ClipData],
                               target_batch_duration: float,
                               context_pre_batch_duration: float,
                               context_post_batch_duration: float,
                               video_background: str) -> List[str | None]:
        """Correct the transcriptions for a contiguous set of clips.

        Args:
            clips_data (Sequence[ClipData]): The clips data.
            target_batch_duration (float): The total length of clips which will be corrected on each LLM operation.
                Except in boundary occasions, the actual length of those clips is guaranteed to be greater than or equal to this argument.
            context_pre_batch_duration (float): The total length of context clips before target clips on each LLM operation.
                Except in boundary occasions, the actual length of those clips is guaranteed to be greater than or equal to this argument.
            context_post_batch_duration (float): The total length of context clips after target clips on each LLM operation.
                Except in boundary occasions, the actual length of those clips is guaranteed to be greater than or equal to this argument.
            video_background (str): Background information about the video.

        Returns:
            List[str | None]: The corrected transcriptions.
                None elements mean that the corresponding clips has no speech or were skipped.
        """
        
        current_start = 0
        transcriptions: List[str] = []
        
        while current_start < len(clips_data):
            # get context_pre
            context_pre: List[ClipData] = []
            context_pre_length = 0
            for i in range(current_start - 1, -1, -1):
                context_pre.insert(0, clips_data[i])
                context_pre_length += clips_data[i].duration
                if context_pre_length >= context_pre_batch_duration:
                    break
            
            # get target clips
            target_clips: List[clips_data] = []
            target_clips_length = 0
            for i in range(current_start, len(clips_data)):
                target_clips.append(clips_data[i])
                target_clips_length += clips_data[i].duration
                if target_clips_length >= target_batch_duration:
                    break
            
            # get context_post
            context_post: List[ClipData] = []
            context_post_length = 0
            for i in range(current_start + 1, len(clips_data)):
                context_post.append(clips_data[i])
                context_post_length += clips_data[i].duration
                if context_post_length >= context_post_batch_duration:
                    break
            
            # correct
            result = self._correct_clip_transcriptions(context_pre, target_clips, context_post, video_background)
            
            # update
            if result is not None:
                for i in range(len(target_clips)):
                    transcriptions.append(result.get(i, None))
            else:
                transcriptions += [None] * len(target_clips)
                print(f'Error occurred when correcting transcriptions for clips [{current_start}, {current_start + len(target_clips)})', file=sys.stderr)
                
            current_start += len(target_clips)

            print(f'\rFinished clips {current_start + len(target_clips)}/{len(clips_data)}')
            
            assert current_start + len(target_clips) == len(transcriptions)
        
        return transcriptions
    
    def _correct_clip_transcriptions(self, context_pre: Sequence[ClipData], targets: Sequence[ClipData], context_post: Sequence[ClipData], video_background: str) -> Dict[int, str] | None:
        """Corrects the transcriptions of the clips in `targets`.

        Args:
            context_pre (Sequence[ClipData]): Data about the clips before `targets`.
            targets (Sequence[ClipData]): Data about the clips to be transcribed.
            context_post (Sequence[ClipData]): Data about the clips after `targets.
            video_background (str): Background information about the video.

        Returns:
            Dict[int, str] | None: The corrected transcriptions.
                The keys correspond to clip indices in `targets`, while the values are the corresponding transcriptions.
                Returning None means there is max_retry_count exceeded.
        """
        
        result = None
        
        for _ in range(self.max_retry_count):
            try:
                correction_prompt = self._build_transcription_correction_prompt(context_pre, targets, context_post, video_background)
                correction_output: str = self.chat_backend([(correction_prompt, True)])
                formatting_prompt = self._build_transcription_formatting_prompt(correction_output, len(targets))
                formatting_output: str = self.chat_backend([formatting_prompt])
                
                transcriptions: Dict[str, str] = json.loads(formatting_output)
                
                target_clip_indices = {len(context_pre) + i for i in range(len(targets))}
                
                if (not target_clip_indices.issuperset(int(key) for key in transcriptions.keys())) \
                    or (not all(isinstance(val, str) for val in transcriptions.values())):
                    raise Exception("Error occurred in transcription correction.")

                result = {int(key) - len(context_pre): transcriptions[key] for key in transcriptions.keys()}
                
                break
            except Exception:
                continue
        
        return result

    def _build_transcription_correction_prompt(
        self,
        context_pre: Sequence[ClipData],
        targets: Sequence[ClipData],
        context_post: Sequence[ClipData],
        video_background: str) -> str:
        """Builds the prompt used to be fed into an LLM and retrieve corrected transcriptions.
        
        The output of the LLM when fed with the returned value is expected to contain the corrected transcriptions,
        but has no specific machine-parsable format.

        Args:
            context_pre (Sequence[ClipData]): Data about the clips before `targets`.
            targets (Sequence[ClipData]): Data about the clips to be transcribed.
            context_post (Sequence[ClipData]): Data about the clips after `targets.
            video_background (str): Background information about the video.

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
Additionally, some clips may not contain any speech at all
(it is also possible that the ASR output is not empty even though the clip contains no speech, so use your discretion).

If you think that a clip does not contain any speech or you really cannot infer a plausible transcription (e.g., when the ASR output is nonsense),
just omit that clip in your output.

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

{video_background}

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
However, please DO indicate the index of the clip that each of your transcriptions corresponds to.
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
Please format the LLM output into JSON so that I can use some simple code to retrieve the transcriptions in a machine-friendly form.

The transcriptions in the LLM output are ordered by clip indices (not necessarily starting from 0 or 1),
and I want you to output the a JSON dictionary where the keys are clip indices and the values are the corresponding transcriptions.
If the transcription of a clip is an empty string, you may either include it (setting the corresponding value to "") in your JSON output or omit it.

For example, if the LLM output is:

    Sure! Here are my transcriptions for clip 15-17:
    
    Clip 15 transcription:
    blah1 blah1 blah1
    
    Clip 110:
    blah2
    
    blah2
    blah2
    
    Clip 16:
    
Then your output should be something like:

    {{
        "15": "blah1 blah1 blah1",
        "16": "",
        "110": "blah2\\n\\nblah2\\nblah2"
    }}

Notice that the clip indices MAY NOT be a list of contiguous numbers.

Now, the LLM output I got is:

{transcription_output}

Please output the formatted transcription ONLY and NOTHING ELSE.
Also, keep the original transcriptions and do not try to correct them even if they seem grammatically incorrect.
"""
