from pathlib import Path

from src.subtitle_generation import DefaultGenerator

generator = DefaultGenerator(quality_preset='low')


generator.generate_subtitles(video_path=Path('/home/trent/Downloads/episode.mp4'),
                             output_path=Path('test-gpt-3.5-turbo.srt'),
                             video_background=\
"""The video consists of two episodes from the Japanese anime Chimpui.
The title of the first episode is "ワンダユウは占い師？"; the title of the second episode is "エリさまは美少女".
There is an introductory screen at the start of each episode, where the title of that episode is shouted out.

Some special terms in Chimpui:

A. Characters:

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

B. Others:
    1. Mahl (マール), the name of the planet that Chinpui and Wanda come from.
    2. Kahou (科法 / かほう). "科法" is the special, advanced and convenient-to-use technologies from the Mahl planet.
    Chinpui and Wanda both uses Kahou.
""",
                             target_language='simplified Chinese',
                             workspace_path=Path('test_workspace'))
