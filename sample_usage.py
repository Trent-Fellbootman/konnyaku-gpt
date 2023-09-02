from src.transcription_correction import TranscriptionCorrector
from src.models.whisper_cloud import WhisperCloud
from src.models.blip_large import BlipLarge
from src.models.gpt4 import GPT4
from src.models.gpt_35_turbo import GPT35Turbo
from src.data_models import ClipData, ClipSetMetadata

from typing import List, Dict
from pathlib import Path

import json
import logging


logging.basicConfig(level=logging.INFO)

corrector = TranscriptionCorrector(
    asr_description=WhisperCloud.get_description(),
    captioner_description=BlipLarge.get_description(),
    chat_backend_corrector=GPT4(max_retry_count=10),
    chat_backend_extractor=GPT35Turbo(max_retry_count=10),
    max_retry_count=10
)

with open('./episode-15-data/transcriptions.json') as f:
    transcriptions = json.loads(f.read())

with open('./episode-15-data/captions.json') as f:
    captions = json.loads(f.read())

with open('./episode-15-data/clips/metadata.json') as f:
    durations = [item.duration for item in ClipSetMetadata.from_json(f.read()).clips_metadata]

assert len(transcriptions) == len(captions) == len(durations)

clips_data = [ClipData(duration, [transcription], caption)
              for transcription, caption, duration in zip(transcriptions, captions, durations)]

transcriptions = corrector.correct_transcriptions(
    clips_data=clips_data[1:21],
    output_file=Path('./episode-15-data/transcriptions_corrected.json'),
    target_batch_duration=80,
    context_pre_batch_duration=20,
    context_post_batch_duration=20,
    video_background=\
"""The video is an episode named "ワンダユウは占い師？" from the Japanese anime Chimpui.

Some characters in Chimpui:

1. 春日エリ (Kasuga Eri), often referred to as "Eri" (Chinese: 惠理), "Eri-chan", "Eri-sama" (Chinese: 惠理大人) and "Kasuga" (Chinese: 春日). She is a boyish girl. She has a crush on her classmate Uchiki and does not want to marry the prince from the Mahl planet at all.
2. Chimpui (チンプイ, Chinese: 芝比). An alien mouse with superpowers.
3. Wanderyu (ワンダユウ, Chinese: 旺达). An alien dog with superpowers.
4. Shou Uchiki (内木翔), often referred to as "Uchiki" (Chinese: 内木). A top student. Eri has a crush on him.
5. Sunemi Koganeyama (小金山スネ美), often referred to as "Sunemi" / "Sunemi-chan" / "Sunemi-san" (Chinese: 诗奈美). A rich girl who likes to brag about things, especially her family's wealth.
6. Masao Oeyama (大江山政男), often referred to as "Oeyama" (Chinese: 大江山). A strong 6-grader who sometimes bullies his classmates, especially Uchiki.
7. Shosei Kitsune (木常小政), often referred to as "Shosei" (Chinese: 小政) or "Kitsune" (Chinese: 木常). Oeyama's sidekick.
8. Hotaru Fujino (藤野ほたる), often referred to as "Hotaru" (Chinese: 小莹). A girlish girl who often dreams about girlish, unrealistic things, like a handsome prince coming to marry her.
9. Lulealv, often referred to as "Lulealv Denka" (Chinese: 吕诺夫殿下) or simply "Denka" (殿下). The prince of the Mahl planet who wishes to marry Eri. However, Eri doesn't want to marry him at all.

Chinpui and Wanda are both from the Mahl (Japanese: マール) planet.
This planet is also referred to as the Mahl planet in English,
or "玛尔星" in Chinese.
""",
    target_language="simplified Chinese"
)