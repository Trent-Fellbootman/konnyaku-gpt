from src.transcription_correction import TranscriptionCorrector
from src.models.whisper import Whisper
from src.models.blip_large import BlipLarge
from src.models.gpt_35_turbo import GPT35Turbo
from src.data_models import ClipData

from typing import List, Dict

import json
import logging
logging.basicConfig(level=logging.INFO)

corrector = TranscriptionCorrector(
    asr_description=Whisper.get_description(),
    captioner_description=BlipLarge.get_description(),
    chat_backend=GPT35Turbo(max_retry_count=3),
    max_retry_count=3
)

with open('./episode-15-data/clips_data.json') as f:
    clips_data: List[Dict] = json.loads(f.read())
    clips_data = [ClipData.from_pytree(clip_data) for clip_data in clips_data]

transcriptions = corrector.correct_transcriptions(
    clips_data=clips_data,
    target_batch_duration=80,
    context_pre_batch_duration=40,
    context_post_batch_duration=40,
    video_background=\
"""This is an episode from the Japanese anime Chimpui.

Some background information of Chimpui:

Chimpui (チンプイ) is a Japanese anime that tells the story of an earth girl named Eri being selected as the pricess of a planet named Mahl.
The Mahl planet sent two cute aliens to earth to persuade Eri to marry the prince of the Mahl planet.

Main characters:

1. Kasuga Eri (春日エリ, often referred to as "Eri", "Eri-chan", "Eri-sama" and "Kasuga"), a 6-grade school girl in Tokyo, Japan.
She is selected as the princess of Mahl, but she actually has a crush on her classmate, Shou Uchiki (内木翔).
Eri behaves like a boy in many ways, like jumping over the fences of her home every time returning from school.
She even fights with the bullies if she see them bullying someone.

2. Chimpui (チンプイ), a young, mouse-like alien from the Mahl planet.
The Mahl planet sent him to live with Eri to persuade her to marry the prince,
but actually, he is more like Eri's friend and seldomly try to talk her into marriage.
Since the Mahl planet has superior technologies over the earth, Chimpui is able to use magic-like tech to help Eri.
Having large ears and a purple body, Chimpui might be mistaken as an elephant.

3. Wanderyu (ワンダユウ), an old dog-like assistant of the prince of Mahl.
He is also able to use magic-like tech from Mahl,
and try to use all measures to persuade (sometimes even coax) Eri into marrying the prince, as soon as possible.
He lives on the Mahl planet, but comes to earth occasionally to visit Eri (and urge Chimpui to be active in persuading Eri).

4. Shou Uchiki (内木翔), Shou is a sixth grader who is good at studies but is weak at sports. He is Eri's very best friend who a crush on him.

5. Sunemi Koganeyama (小金山スネ美), she is Eri's rich friend with a fox-like face. She likes to brag about her family's wealth.

6. Masao Oeyama (大江山政男), He is a strong 6th grader who is good at sports and he bullies Shou.

7. Shosei Kitsune (木常小政), A 6th grader who is absolutely short for his age and is Masao's sidekick.

8. Hotaru Fujino (藤野ほたる), a 6th grader and one of Eri's best friends. She often dreams about unrealistic things,
like a handsome prince coming to marry her.

9. Lulealv, the prince of Mahl whose face is never shown. Often referred to as "Lulealv Denka" or simply "Denka" (殿下).
""",
    target_language=None
)

with open('./episode-15-transcriptions.json', 'x') as f:
    f.write(json.dumps([transcription if transcription is not None else '' for transcription in transcriptions], indent=4))