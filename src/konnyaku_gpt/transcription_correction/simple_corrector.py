import json
from typing import Sequence, Dict, List, Any, Tuple
from src.data_models import ClipData
from .transcription_corrector import TranscriptionCorrector
from ..models.chat_completion import ChatCompletionService


class SimpleCorrector(TranscriptionCorrector):
    
    """A simple corrector where only two LLM calls are involved in a transcription correction operation
    (one for correction and another for transcription extraction from natural language)
    """

    def __init__(self, correction_backend: ChatCompletionService, extraction_backend: ChatCompletionService, max_try_count: int=3):
        
        """Constructor.

        Args:
            correction_backend (ChatCompletionService): The model service used for correction.
            extraction_backend (ChatCompletionService): The model service used for transcription extraction from natural language.
            max_try_count (int): The maximum number to retry on each group of clips. If reached, exception will be raised.
        """
        
        super().__init__()

        self.correction_backend = correction_backend
        self.extraction_backend = extraction_backend
        self.max_retry_count = max_try_count
    
    # override
    def correct_transcriptions(self, clips_data: Sequence[ClipData], video_background: str, auxiliary_information: str, target_language: str | None = None) -> Sequence[str]:
        for _ in range(self.max_retry_count):
            try:
                correction_prompt = self._build_transcription_correction_prompt(
                    clips_data=clips_data,
                    auxiliary_information=auxiliary_information,
                    video_background=video_background,
                    target_language=target_language
                )
                correction_result = self.correction_backend([(correction_prompt, True)])
                extraction_prompt = self._build_transcription_formatting_prompt(
                    transcription_output=correction_result,
                    n_clips=len(clips_data),
                    target_language=target_language
                )
                extraction_result = self.extraction_backend([(extraction_prompt, True)])

                transcriptions: Dict[str, str] = json.loads(extraction_result)
                assert set(range(len(clips_data))).issuperset(int(key) for key in transcriptions.keys()), "Invalid clip indices detected in formatted transcriptions!"
                assert isinstance(transcriptions, Dict), "Unexpected JSON structure from extraction output!"
                assert all(isinstance(val, str) for val in transcriptions.values()), "Unexpected JSON structure from extraction output!"

                return [transcriptions.get(str(i), '') for i in range(len(clips_data))]
                
            except Exception as e:
                continue
        
        raise Exception(f'Max retry count of {self.max_retry_count} reached!')

    def _build_transcription_correction_prompt(
        self,
        clips_data: Sequence[ClipData],
        auxiliary_information: str,
        video_background: str,
        target_language: str | None=None) -> str:
        """Builds the prompt used to be fed into an LLM and retrieve corrected transcriptions.
        
        The output of the LLM when fed with the returned value is expected to contain the corrected transcriptions,
        but has no specific machine-parsable format.

        Args:
            clips_data (Sequence[ClipData]): The clips whose translations are to be corrected.
            auxiliary_information (str): Any auxiliary information like ASR & image-to-text model quirks.
            video_background (str): Background information about the video.
            target_language (str | None): The target language.

        Returns:
            str: The prompt ready to be fed into an LLM.
        """
        
        def format_clip(index: int, transcription: str, caption: str) -> str:
            return \
f"""Clip {index}:
(Inaccurate) transcription: {transcription}
(Inaccurate) description of an arbitrarily picked frame: {caption}"""
        
        clip_args = [(i, '\n'.join(item.audio_transcriptions_raw), item.screenshot_description) for i, item in enumerate(clips_data)]
        
        clips_data_part = ('\n' * 2).join(format_clip(*args) for args in clip_args)

        return \
f"""I am trying to create subtitles for a video.
I split the video into many clips and used an Automatic Speech Recognition (ASR) model to transcribe the audio of each clip.
I also selected an arbitrary frame from each clip and used an image-to-text model to create a description for it.

Now, here are the information of the clips
(0 indexed, indices correspond to the order in which the clips appear in the video;
an empty transcription means that the ASR model did not detect any speech in the corresponding clip):

{clips_data_part}

Here is some additional information which you may find helpful:

<additional-information-start>
{auxiliary_information}
<additional-information-end>

Here is some background information about the video from which the clips are extracted
(however, the clips I gave you are not all the clips in the video):

<video-background-start>
{video_background}
<video-background-end>

The ASR and image-to-text model are very inaccurate.
Therefore, I need you to use logical reasoning (and your imagination when information is insufficient) to infer what is going on in the video,
and then correct the transcriptions so that they look COHERENT and NATURAL, and ARE LOGICALLY CONNECTED.
{f"I also want you to translate and provide your corrected transcriptions in {target_language}."if target_language is not None else ''}

Although there are likely to be multiple characters speaking, you DO NOT need to separate the speech of each speaker.
Just provide the combined audio transcriptions for each clip
(however, you may use quotations to indicate the partitions between multiple speeches in a transcription).

Make sure to look for potential cases where special terms were not correctly identified by the ASR model,
and address them properly. You should look at the background information to know the special terms.

Some keynotes:

1. A transcription may contain no speech, incomplete / complete speech by one character, or combined speech of multiple characters.
2. ASR and image-to-text models may have special quirks. Pay attention to their special behavior.
3. The ASR model has no access to background information and may misrecognize special terms.
If you see a phrase that look strange or does not fit into its context, there is a good chance that it's a misrecognized special terms.
Sometimes ASR will recognize the presence of the special term, but will fail to spell it correctly.
4. The clips may include multiple scenes. There may also be transition scenes.

You should provide your revised {"translated " if target_language is not None else ''}transcriptions for all the clips.
DO NOT include original ASR outputs or image-to-text outputs.
However, please DO indicate the index of the clip that each of your transcriptions corresponds to.

You should output transcriptions all the clips I ask you to transcribe even if there are many.

Please make sure to address the special terms (especially character names) correctly.
For example, if the video is a Doraemon episode and the ASR model outputs "どかえ保",
there's a large probability that the correct transcription is ドラえもん
(remember, the ASR model is likely to recognize syllables incorrectly and then further interpret them as incorrect words),
and you should output that (or "哆啦A梦" if I ask you to translate into Chinese).
YOU MUST CONVERT THE ASR OUTPUT INTO SYLLABLES, CORRECT THE INCORRECT SYLLABLES, AND INTERPRET THE SYLLABLES CORRECTLY.
Also, take into account the specific behavior of the ASR & image-to-text models.

Pay attention to the quirks of ASR and image-to-text model.

Please make sure that your corrected transcriptions look NATURAL to native speakers, make LOGICAL sense and are plausible given the video's background information i provided to you.
They should not look like human-unintelligible nonsense produced by low-quality machine translation service.
If a transcription really does not fit into its context and you cannot infer the correct transcription,
you should look at its context and use your imagination to write a transcription yourself, instead of keeping the transcription as-is.
You should try to identify potential occurrences of special terms (especially character names) and spell and address them correctly according to the background information.

Remember, you should {f"translate the transcriptions into {target_language}." if target_language is not None else "keep the transcriptions in their original language."}

Keep in mind that your priority is to provide LOGICALLY-CONNECTED transcriptions that LOOK NATURAL AND COHERENT TO NATIVE SPEAKERS, NOT TO TRY TO INTERPRET THE ASR OUTPUTS.
In case you really cannot infer the correct transcription for a clip, you should look at its context and write a transcription by yourself.

Provide your revised transcriptions ONLY and NOTHING ELSE.
"""

    def _build_transcription_formatting_prompt(self, transcription_output: str, n_clips: int, target_language: str | None) -> str:
        """Builds the prompt used to be fed into an LLM and retrieve corrected, formatted transcriptions.

        Args:
            transcription_output (str): LLM's output containing the transcriptions.
            n_clips (int): The (expected) number of clips included in `transcription_output`.
            target_language (str | None): The target language.

        Returns:
            str: The prompt ready to be fed into the LLM.
        """
        
        return \
f"""I used an LLM to correct transcriptions for a number of audio clips.
Please format the LLM output into JSON so that I can use some simple code to retrieve the transcriptions in a machine-friendly form.

The transcriptions in the LLM output are ordered by clip indices (not necessarily starting from 0 or 1),
and I want you to output the a JSON dictionary where the keys are clip indices and the values are the corresponding transcriptions.
If the transcription of a clip is an empty string, you may either include it (setting the corresponding value to "") in your JSON output or omit it.

Notice that the clip indices MAY NOT be a list of contiguous numbers.

Even if the LLM separated the transcriptions into speech of multiple speaker,
DO NOT output the transcriptions of each speaker separately.
Instead, OUTPUT THE COMBINED TRANSCRIPTION FOR EACH CLIP.

Now, the LLM output I got is:

{transcription_output}

{
f'''The LLM should have translated the transcriptions into {target_language}. Please output the TRANSLATIONS instead of the original transcriptions (if they are also provided).
''' if target_language is not None else
'''The LLM may have translated the transcriptions, but even if this is the case, please output the transcriptions in their ORIGINAL language.
'''
}

If you think some transcriptions do not make sense (e.g., random UTF-8 emojis), you may try to correct them so that it looks natural and plausible,
or simply omit them.

{'' if target_language is None else f'If the LLM output is not in {target_language}, you should also translate the transcriptions into {target_language}.'}

Please output the JSON-form transcriptions ONLY and NOTHING ELSE.

As an example, if the LLM output is:

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

Remember, the transcriptions in your output should be in {target_language if target_language is not None else "their original language"}.
"""
