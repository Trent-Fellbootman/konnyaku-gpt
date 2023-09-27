import logging
import json
from typing import Sequence, List, Dict, Any, Tuple, Callable
from pathlib import Path

from ..data_models import ClipData
from .many_clips_transcription_corrector import ManyClipsTranscriptionCorrector
from ..transcription_correction.transcription_corrector import TranscriptionCorrector


class ManyClipsBatchedCorrector(ManyClipsTranscriptionCorrector):
    
    """A clip corrector that is able to correct a large number of clips by
    grouping them into batches (batches may intersect) and applying another "group corrector" for each batch.
    
    Since this corrector deals with large number of clips, a caching mechanism is added so that correction operation can be paused and resumed.
    """
    
    def __init__(self, group_corrector: TranscriptionCorrector, max_retry_count: int=3) -> None:
        """Constructor.

        Args:
            group_corrector (TranscriptionCorrector): The group corrector that will be applied to each batch.
            max_retry_count (int): The maximum number to retry on each group of clips. If reached, the clip group will be skipped and no transcription will be produced.
        """
        
        super().__init__()
        
        self.group_corrector = group_corrector
        self.max_retry_count = max_retry_count
    
    # override
    def correct_transcriptions(self,
                               clips_data: Sequence[ClipData],
                               video_background: str,
                               auxiliary_information: str,
                               target_language: str | None,
                               cache_path: Path | None,
                               min_target_clips_length: float=40,
                               min_pre_context_length: float | None=None,
                               min_post_context_length: float | None=None,
                               group_completion_callback: Callable[[], None]=lambda *args, **kwargs: None) -> Sequence[str]:
        """Correct the transcriptions.
        
        In a batch, target clips are those whose corrected transcriptions will be actually used;
        pre- and post- context clips are to provide contextual information only and their corrected transcriptions will be discarded.

        Args:
            clips_data (Sequence[ClipData]): The clips whose translations are to be corrected.
            video_background (str): The background information of the video that the clips come from.   
            auxiliary_information (str): Any auxiliary information like ASR & image-to-text model quirks.
            target_language (str | None): None if transcriptions should not be translated, the target language of translation otherwise.
            min_target_clips_length (float): The minimum length of the target clips in each batch.
            min_pre_context_length (float | None, optional): The minimum length of the pre-target context clips in each batch. "None" means 50% of `target_clips_length`. Defaults to None.
            min_post_context_length (float | None, optional): The minimum length of the post-target context clips in each batch. "None" means 50% of `target_clips_length`. Defaults to None.
            cache_path (Path | None, optional): The transcription output filepath. "None" means no cache file. Defaults to None.
                If a cache file is specified, that file will be used to store the partial results when the corrector is paused.
            group_completion_callback (Callable[[], None], optional): A callback function to be called when each batch is completed.

        Returns:
            Sequence[str]: The corrected transcriptions.
        """

        if min_pre_context_length is None:
            min_pre_context_length = min_target_clips_length * 0.5
        
        if min_post_context_length is None:
            min_post_context_length = min_target_clips_length * 0.5
        
        # compile clip groups
        current_start = 0
        # [(pre context start index, pre context clips count, target clips count, post context clips count)]
        groups: List[Tuple[int, int, int, int]] = []
        
        while current_start < len(clips_data):
            # get context_pre
            context_pre: List[ClipData] = []
            context_pre_length = 0
            for i in range(current_start - 1, -1, -1):
                context_pre.insert(0, clips_data[i])
                context_pre_length += clips_data[i].duration
                if context_pre_length >= min_pre_context_length:
                    break
            
            # get target clips
            target_clips: List[clips_data] = []
            target_clips_length = 0
            for i in range(current_start, len(clips_data)):
                target_clips.append(clips_data[i])
                target_clips_length += clips_data[i].duration
                if target_clips_length >= min_target_clips_length:
                    break
            
            # get context_post
            context_post: List[ClipData] = []
            context_post_length = 0
            for i in range(current_start + len(target_clips), len(clips_data)):
                context_post.append(clips_data[i])
                context_post_length += clips_data[i].duration
                if context_post_length >= min_post_context_length:
                    break
            
            # create group
            groups.append((current_start - len(context_pre), len(context_pre), len(target_clips), len(context_post)))
            assert groups[-1][0] + len(context_pre) == current_start
                
            current_start += len(target_clips)
            assert groups[-1][0] + len(context_pre) + len(target_clips) == current_start

        assert sum(group[2] for group in groups) == len(clips_data)
        
        # see which groups have been processed and which have not
        if cache_path is not None:
            cache_path.touch()
            
            try:
                with open(cache_path, 'r') as f:
                    cached_transcriptions = json.load(f)
                
                # must be a dict
                assert isinstance(cached_transcriptions, Dict)
                # keys must be subset of all possible group indices
                assert set(int(key) for key in cached_transcriptions.keys()).issubset(range(len(groups)))
                
                for key, value in cached_transcriptions.items():
                    # transcriptions for a group must be a list
                    assert isinstance(value, List)
                    # each transcription must be a str
                    assert all(isinstance(t, str) for t in value)
                    # number of transcriptions must equal that of the target clips in that group
                    assert len(value) == groups[int(key)][2]
                
                transcriptions = {int(key): value for key, value in cached_transcriptions.items()}
                    
            except Exception as e:
                # if any error is found in the cache, invalidate it.
                # {group index: {transcriptions of target clips in that group}}
                transcriptions: Dict[int, List[str]] = {}
        else:
            transcriptions: Dict[int, List[str]] = {}
        
        # correct transcriptions
        unprocessed_groups = set(range(len(groups))).difference(transcriptions.keys())

        total_unprocessed_groups = len(unprocessed_groups)

        while len(unprocessed_groups) > 0:
            logging.info(f'{len(unprocessed_groups)}/{total_unprocessed_groups} groups remaining')
            # get group to be processed
            group_index = next(iter(unprocessed_groups))
            group = groups[group_index]
            start_index, context_pre_clip_count, target_clips_count, post_context_clip_count = group
            
            # get pre context clips
            pre_context_clips = clips_data[start_index:start_index + context_pre_clip_count]
            # get target clips
            target_clips = clips_data[start_index + context_pre_clip_count:start_index + context_pre_clip_count + target_clips_count]
            # get post context clips
            post_context_clips = clips_data[start_index + context_pre_clip_count + target_clips_count:start_index + context_pre_clip_count + target_clips_count + post_context_clip_count]

            # correct the clips
            corrected_transcriptions = None
            for _ in range(self.max_retry_count):
                try:
                    corrected_transcriptions = self.group_corrector.correct_transcriptions(
                        clips_data=list(pre_context_clips) + list(target_clips) + list(post_context_clips),
                        video_background=video_background,
                        auxiliary_information=auxiliary_information,
                        target_language=target_language
                    )
                    break
                except Exception as e:
                    continue
            
            if corrected_transcriptions is None:
                # failure, skip these clips
                # TODO: caching behavior? doing this would make these clips never be tried again!
                transcriptions[group_index] = [''] * target_clips_count
                
                logging.warning(f'Max retry count reached; group {group_index} skipped.')
            else:
                # success, add corrected transcriptions to transcriptions
                transcriptions[group_index] = corrected_transcriptions[context_pre_clip_count:context_pre_clip_count + target_clips_count]
                logging.info(f'Group {group_index} corrected successfully.')
            
            assert len(transcriptions[group_index]) == target_clips_count

            # save the transcriptions to cache file
            if cache_path is not None:
                with open(cache_path, 'w') as f:
                    json.dump(transcriptions, f, indent=4, ensure_ascii=False)

            unprocessed_groups.remove(group_index)
            
            # call callback
            group_completion_callback()

        # combine the trancriptions from each group
        combined_transcriptions = []
        for i in range(len(groups)):
            combined_transcriptions += transcriptions[i]
        
        return combined_transcriptions
