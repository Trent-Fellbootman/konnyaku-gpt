import json
from typing import Sequence, Dict, List, Any, Tuple, Callable
from src.data_models import ClipData
from .transcription_corrector import TranscriptionCorrector
from ..models.chat_completion import ChatCompletionService


class MultiRoundCorrector(TranscriptionCorrector):
    
    """A multi-round corrector corrects transcriptions in multiple rounds of conversations.
    
    Concretely, the process of correcting a set of transcriptions are as follows:
    
    1. Original clips data, video background and other auxiliary information are used to analyze what is going on in each clip.
    2. A chat completion service tries to identify transcription errors and suggest fixes based on the plot analysis.
    3. A chat completion service tries to correct the transcriptions and produce the corrected transcriptions.
    4. If the transcriptions need to be translated, a chat completion service tries to translate the transcriptions. Otherwise, do nothing.
    5. A chat completion service is used to convert the revised transcriptions into machine-readable form.
    
    Except the final step, the process happens as a continuous, multi-round conversation.
    """

    def __init__(self,
                 plot_analysis_backend: ChatCompletionService,
                 fix_suggestion_backend: ChatCompletionService,
                 fix_application_backend: ChatCompletionService,
                 translation_backend: ChatCompletionService,
                 transcription_extraction_backend: ChatCompletionService,
                 max_retry_count: int=3):
        """Constructor.

        Args:
            plot_analysis_backend (ChatCompletionService): Chat completion service used to analyze what is going on in each clip.
            fix_suggestion_backend (ChatCompletionService): Chat completion service used to suggest fixes based on the plot analysis.
            fix_application_backend (ChatCompletionService): Chat completion service used to correct the transcriptions and produce the corrected transcriptions, based on the proposed fixes.
            translation_backend (ChatCompletionService): Chat completion service used to translate the transcriptions.
            transcription_extraction_backend (ChatCompletionService): Chat completion service used to convert the revised transcriptions into machine-readable form.
            max_retry_count (int, optional): The maximum number to retry on each group of clips. Defaults to 3.
        """

        super().__init__()

        self.plot_analysis_backend = plot_analysis_backend
        self.fix_suggestion_backend = fix_suggestion_backend
        self.fix_application_backend = fix_application_backend
        self.translation_backend = translation_backend
        self.transcription_extraction_backend = transcription_extraction_backend
        self.max_retry_count = max_retry_count
    
    # override
    def correct_transcriptions(self, clips_data: Sequence[ClipData], video_background: str, auxiliary_information: str, target_language: str | None = None) -> Sequence[str]:
        for _ in range(self.max_retry_count):
            try:
                def format_clip(index: int, transcription: str, caption: str) -> str:
                    return \
f"""Clip {index}:
(Inaccurate) transcription: {transcription}
(Inaccurate) description of an arbitrarily picked frame: {caption}"""
        
                clip_args = [(i, '\n'.join(item.audio_transcriptions_raw), item.screenshot_description) for i, item in enumerate(clips_data)]
                clips_data_part = ('\n' * 2).join(format_clip(*args) for args in clip_args)
                
                chat_history = []
                
                # step 1: plot analysis
                analysis_prompt = \
f"""I am creating subtitles for a video. I splitted it into clips and used speech recognition to create transcriptions for each clip. I also used an image-to-text model to create a description of an arbitrarily picked frame in each clip.

The transcriptions and frame descriptions are:

<clips-information-start>
{clips_data_part}
<clips-information-end>

Notice that some clips might have no speech at all.

For your reference, here are some additional information:

<additional-information-start>
{auxiliary_information}
<additional-information-end>

Here are some background information about the video which the clips come from (however, the clips I gave you are not all the clips):

<background-information-start>
{video_background}
<background-information-end>

Now, please analyze the big picture of what is going on, then analyze the activity of each clip based on that. Make sure to indicate how each clip relate to other clips. Notice that some transcriptions may contain multiple speeches spoken by multiple persons. Also the frame descriptions can be complete nonsense sometimes, so beware if you see a frame description that doesn't make sense. You should take into account the transcriptions, frame descriptions, and your knowledge about Japanese culture.

Notice that the transcriptions & frame descriptions can be very inaccurate and sometimes even misleading. Hence, you should value your logical reasoning (and even imagination) more than them. You are encouraged to also analyze which transcriptions / frame descriptions might be complete nonsense and misleading.

Also, change of scenes and transitory scenes may occur in the clips. The clips may involve multiple scenes with somewhat disconnected stories.

You DO NOT need to assign activities to specific characters.

Just provide the analysis; DO NOT include the original transcriptions & frame descriptions.

Make sure to take into account the special behavior of the speech recognition model and the image-to-text model. They may have special quirks that produce misleading outputs.

Pay attention to the special behavior of the speech recognition model and the image-to-text model. Their quirks may have resulted in incorrect and even misleading outputs which you should identify and ignore, instead of being affected.

OUTPUT THE SUMMARY OF STORYLINE FIRST; USE INFORMATION FROM OTHER CLIPS AND THE GENERAL PLOT TO HELP YOU DETERMINE WHAT IS GOING ON IN EACH CLIP. SOME CLIPS MAY NOT MAKE SENSE ON THEIR OWN."""

                chat_history.append((analysis_prompt, True))
                analysis_result = self.plot_analysis_backend(chat_history)
                chat_history.append((analysis_result, False))
                
                # step 2: suggest fixes
                fix_suggestion_prompt = \
"""Looking at the background information and your analysis, please identify possible errors in the transcriptions, infer what the erroneous words / phrases / transcriptions might actually be, and suggest a replacement that fits into the context and makes sense to you for each of them.

You may want to look back at the background information and identify special terms that may occur. If you see a phrase that looks especially strange or does not fit into its context, it is likely a misrecognized special term, like a character name. Also, sometimes speech recognition can recognize the presence of a special term, but because it has no access to background information, the result is likely a misspelled special term. You should suggest the correct replacement for misrecognized or misspelled special terms as well.

You may want to pay attention to special behavior of the speech recognition model and the image-to-text model. They may have special quirks that produce incorrect and even misleading information."""
                chat_history.append((fix_suggestion_prompt, True))
                fix_suggestion_result = self.fix_suggestion_backend(chat_history)
                chat_history.append((fix_suggestion_result, False))

                # step 3: apply fixes
                fix_application_prompt = \
"""Now, applying the corrections you suggested, please provide a better version of the transcriptions for all the clips. Please also correct any errors that you did not identify previously (pay extra attention to potential special terms). Provide the corrected transcriptions ONLY."""
                chat_history.append((fix_application_prompt, True))
                fix_application_result = self.fix_application_backend(chat_history)
                chat_history.append((fix_application_result, False))

                # step 4: translate (if applicable)
                if target_language is not None:
                    translation_prompt = \
"""Now, please translate the translations into simplified Chinese. You should take into account all information you have and all analysis you have done to ensure that the translated transcriptions ARE LOGICALLY CONNECTED and SOUND NATURAL TO NATIVE SPEAKERS. Also, make sure to translate special terms correctly; you may need to refer to the background information to see how to translate each special term.

You should correct the phrases that do not make sense or do not fit into their context if there are still such phrases after your correction. In case you really cannot infer what the correct transcription is, you should use your imagination and contextual information to write a transcription by yourself. The priority is to make the transcriptions sound NATURAL and look LOGICALLY CONNECTED, NOT to translate the speech recognition outputs as is. Speech recognition outputs can be very inaccurate.

Output the translated transcriptions ONLY and NOTHING ELSE. You should include transcriptions for all the clips, including those you did not modify."""
                    chat_history.append((translation_prompt, True))
                    translation_result = self.translation_backend(chat_history)
                    chat_history.append((translation_result, False))

                    final_transcriptions = translation_result
                else:
                    final_transcriptions = fix_application_result
                
                # step 5: convert natural language transcriptions into JSON
                transcription_extraction_prompt = \
f"""I used an LLM to correct transcriptions for a number of audio clips.
Please format the LLM output into JSON so that I can use some simple code to retrieve the transcriptions in a machine-friendly form.

The transcriptions in the LLM output are ordered by clip indices (not necessarily starting from 0 or 1),
and I want you to output the a JSON dictionary where the keys are clip indices and the values are the corresponding transcriptions.
If the transcription of a clip is empty (e.g., marked as "empty-speech", "no-speech", etc.), you may either include it (setting the corresponding value to "") in your JSON output or omit it.

Notice that the clip indices MAY NOT be a list of contiguous numbers.

Even if the LLM separated the transcriptions into speech of multiple speaker,
DO NOT output the transcriptions of each speaker separately.
Instead, OUTPUT THE COMBINED TRANSCRIPTION FOR EACH CLIP.

Now, the LLM output I got is:

<LLM-output-start>
{final_transcriptions}
<LLM-output-end>

{
f'''The LLM should have translated the transcriptions into {target_language}. Please output the TRANSLATIONS instead of the original transcriptions (if they are also provided).
''' if target_language is not None else
'''The LLM may have translated the transcriptions, but even if this is the case, please output the transcriptions in their ORIGINAL language.
'''
}

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

                # no context because this task is easy
                transcription_extraction_result = self.transcription_extraction_backend([(transcription_extraction_prompt, True)])

                # parse the transcriptions into JSON
                parsed_transcriptions: Dict[str, str] = json.loads(transcription_extraction_result)
                assert set(range(len(clips_data))).issuperset(int(key) for key in parsed_transcriptions.keys()), "Invalid clip indices detected in formatted transcriptions!"
                assert isinstance(parsed_transcriptions, Dict), "Unexpected JSON structure from extraction output!"
                assert all(isinstance(val, str) for val in parsed_transcriptions.values()), "Unexpected JSON structure from extraction output!"

                return [parsed_transcriptions.get(str(i), '') for i in range(len(clips_data))]

            except Exception as e:
                continue
        
        raise Exception('Max retry count of {} reached!'.format(self.max_retry_count))
