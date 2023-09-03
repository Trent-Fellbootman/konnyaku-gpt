import logging
logging.basicConfig(level=logging.INFO)

import json
from pathlib import Path
from typing import List, Tuple, Any, Dict
from pprint import pprint

from src.data_models import ClipData, ClipSetMetadata
from src.many_clips_transcription_correction import ManyClipsBatchedCorrector
from src.transcription_correction import MultiRoundCorrector, SimpleCorrector

from src.models.gpt.openai_gpt_server import OpenAiGptServer
from src.models.gpt.gpt_35_turbo import GPT35Turbo
from src.models.gpt.gpt4 import GPT4

from src.models.dummy_chat_completion_service import DummyChatCompletionService as FakeGPT


data_path = Path('./episode-15-data')

with open(data_path / 'captions.json', 'r') as f:
    captions = json.loads(f.read())

with open(data_path / 'transcriptions.json', 'r') as f:
    transcriptions = json.loads(f.read())

with open('episode-15-data/clips/metadata.json', 'r') as f:
    durations = [item.duration for item in ClipSetMetadata.from_json(f.read()).clips_metadata]

assert len(durations) == len(transcriptions) == len(captions)

data = list(zip(durations, transcriptions, captions))

clips_data = [ClipData(duration, [transcription], caption) for duration, transcription, caption in data]

gpt_server = OpenAiGptServer()
gpt_35_16k = GPT35Turbo(gpt_server, context_length='16k')

corrector = ManyClipsBatchedCorrector(
    group_corrector=MultiRoundCorrector(
        gpt_35_16k,
        gpt_35_16k,
        gpt_35_16k,
        gpt_35_16k,
        gpt_35_16k
    ),
    max_retry_count=3
)

current_total_cost = 0

def callback():
    global current_total_cost
    logging.info(f'Cost on this group: ${gpt_server.money_spent["total"] - current_total_cost: .2e}')
    current_total_cost = gpt_server.money_spent["total"]
    logging.info(f'Cumulative cost: ${current_total_cost: .2e}')

corrected_transcriptions = corrector.correct_transcriptions(
    clips_data=clips_data,
    video_background=\
"""The video consists of two episodes from the Japanese anime Chimpui.
The title of the first episode is "ワンダユウは占い師？"; the title of the second episode is "エリさまは美少女".
There is an introductory screen at the start of each episode, where the title of that episode is shouted out.

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
    auxiliary_information=\
"""The speech recognition model used to produce the original transcriptions is named "Whisper".
It is an ASR model for Japanese transcription.
The model should not be assumed to have world knowledge, and its outputs are likely to be very inaccurate.
Problems in its outputs may include but are not limited to:

1. Incorrect recognition of kanas, e.g., recognizing "ぶ" as "ぷ".
2. Failure to identify prolonged kanas, e.g., recognizing "えい" as "え".
3. Incorrect semantics. These include recognizing hiraganas as katakanas and vice versa,
4. Failure to address special terms properly.
5. SPECIAL PROBLEM: the model occasionally produces non-Japanese output, like English, Korean, and even utf-8 stickers.
6. SPECIAL PROBLEM: the model may incorrectly output something like "ご視聴ありがとうございました" when there is actually no speech.
PAY EXTRA ATTENTION WHEN SEEING "ご視聴ありがとうございました" BECAUSE THERE IS LIKELY TO BE NO SPEECH AT ALL.

translating kanas to wrong Kanjis, incorrect semantic grouping & punctuation (e.g., "はい、そうです。" v.s. "配送です。"), etc.

The model used to produce frame descriptions is named "blip-image-captioning-large".
Its output is likely to be very inaccurate.
Possible problems include but are not limited to:

1. Incomplete description of images. This include failure to detect certain objects (especially when the object is small),
and lack of detailed object descriptions (e.g., "a dog" v.s. "a large, brown dog with collars on its neck").
2. Recignizing one object as something else, e.g., seeing an elliptic spaceship as a ball.
These problems are especially prominent when the input image is not a photograph.
3. Human-specific problems: The model often fail to identify the age and gender of a person correctly.
For example, when the model says there is "a man", it may actually be a girl who is still in primary school.
These problems are especially prominent when the input image is not a photograph.
4. Incorrect identification of image style. E.g., a photo might be recognized as a painting, and vice versa.

Test results show that the model performs well on photos only; very inaccurate results are seen when the model is fed with anime screenshots.
""",
    target_language="Simplified Chinese",
    min_target_clips_length=40,
    min_pre_context_length=10,
    min_post_context_length=10,
    cache_filepath=Path('tmp.json'),
    group_completion_callback=callback
)

pprint(corrected_transcriptions)
