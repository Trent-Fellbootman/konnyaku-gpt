from src.transcription_correction import TranscriptionCorrector
from src.models.whisper_cloud import WhisperCloud
from src.models.blip_large import BlipLarge
from src.models.gpt4 import GPT4
from src.models.gpt_35_turbo import GPT35Turbo
from src.data_models import ClipData, ClipSetMetadata

from typing import List, Dict

import json
import logging
logging.basicConfig(level=logging.INFO)

corrector = TranscriptionCorrector(
    asr_description=WhisperCloud.get_description(),
    captioner_description=BlipLarge.get_description(),
    chat_backend_corrector=GPT35Turbo(max_retry_count=10),
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
    clips_data=clips_data,
    target_batch_duration=80,
    context_pre_batch_duration=20,
    context_post_batch_duration=20,
    video_background=\
"""This is an episode from the Japanese anime Chimpui.

Some background information of Chimpui:

Chimpui (チンプイ) is a Japanese anime that tells the story of an earth girl named Eri being selected as the pricess of a planet named Mahl.
The Mahl planet sent two cute aliens to earth to persuade Eri to marry the prince of the Mahl planet.

Main characters:

1. Kasuga Eri (春日エリ (Chinese: 春日惠理), often referred to as "Eri" (Chinese: 惠理), "Eri-chan", "Eri-sama" (Chinese: 惠理大人) and "Kasuga" (Chinese: 春日)), a 6-grade school girl in Tokyo, Japan.
She is selected as the princess of Mahl, but she actually has a crush on her classmate, Shou Uchiki (内木翔).
Eri behaves like a boy in many ways, like jumping over the fences of her home every time returning from school.
She even fights with the bullies if she see them bullying someone.

2. Chimpui (チンプイ, Chinese: 芝比), a young, mouse-like alien from the Mahl planet.
The Mahl planet sent him to live with Eri to persuade her to marry the prince,
but actually, he is more like Eri's friend and seldomly try to talk her into marriage.
Since the Mahl planet has superior technologies over the earth, Chimpui is able to use magic-like tech to help Eri.
Having large ears and a purple body, Chimpui might be mistaken as an elephant.
One of his habits is to shout "Chinpui" before using Malu's tech.

3. Wanderyu (ワンダユウ, Chinese: 旺达), an old dog-like assistant of the prince of Mahl.
Being an elderly alien, he is often addressed as "Grandpa Wanderyu" by Chimpui.
He is also able to use magic-like tech from Mahl,
and try to use all measures to persuade (sometimes even coax) Eri into marrying the prince, as soon as possible.
He lives on the Mahl planet, but comes to earth occasionally to visit Eri (and urge Chimpui to be active in persuading Eri).
One of his habits is to shout "Wandayo------" before using Malu's tech.
Since Kasuga Eri is seen as the future princess, he usually address Eri as "Eri-sama" (English: Your highness, Eri, or Chinese: 惠理大人) to show his reverence.

4. Shou Uchiki (内木翔), Shou is a sixth grader who is good at studies but is weak at sports. He is Eri's very best friend who a crush on him.
He is often referred to as "Uchiki" (Chinese: 内木).

5. Sunemi Koganeyama (小金山スネ美), she is Eri's rich friend with a fox-like face. She likes to brag about her family's wealth.
She is often referred to as "Sunemi" (Chinese: 诗奈美).

6. Masao Oeyama (大江山政男), He is a strong 6th grader who is good at sports and he bullies Shou.
He is often referred to as "Oeyama" (Chinese: 大江山).

7. Shosei Kitsune (木常小政), A 6th grader who is absolutely short for his age and is Masao's sidekick.
He is often referred to as "Shosei" (Chinese: 小政) or "Kitsune" (Chinese: 木常).

8. Hotaru Fujino (藤野ほたる), a 6th grader and one of Eri's best friends. She often dreams about unrealistic things,
like a handsome prince coming to marry her. She is often referred to (sometimes by herself) as "Hotaru" (Chinese: 小莹).

9. Lulealv, the prince of Mahl whose face is never shown. Often referred to as "Lulealv Denka" (Chinese: 吕诺夫殿下) or simply "Denka" (殿下).

Other important information:

Chinpui and Wanda are both from the Mahl (Japanese: マール) planet.
This planet is also referred to as the Mahl planet in English,
or "玛尔星" in Chinese.

Make sure to address special terms like "チンプイ", "エリ", "ワンダ", "スネ美", "マール", "ほたる", etc.
""",
    target_language="simplified Chinese"
)

with open('./episode-15-transcriptions.json', 'x') as f:
    f.write(json.dumps([transcription if transcription is not None else '' for transcription in transcriptions], indent=4))