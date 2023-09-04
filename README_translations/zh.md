# KonnyakuGPT: 使用多模态GPT实现更准确的动画字幕生成

[English](../README.md) [日本語](ja.md)

README的中文版本是通过ChatGPT翻译英文版本得到，有些地方可能不太准确。

KonnyakuGPT是一个由人工智能驱动的高质量、易于使用的字幕生成器，主要面向日本动画。

## 主要特点

1. **易于使用**：仅需一个函数调用，即可从视频生成高质量字幕文件。
2. **高质量**（取决于您的预算）：使用成本约为每20分钟视频4美元的“中等”质量预设，KonnyakuGPT能够准确处理80%的视频专用术语（角色名称等），并生成构建逻辑故事的字幕。
   **由KonnyakuGPT生成的字幕使不懂日语的观众能够观看日本动画并理解情节，无需猜测发生了什么。**
3. **翻译支持**：KonnyakuGPT 可以为任何语言创建翻译字幕。
4. **暂停和恢复**：在生成字幕时定期保存进度。
   即使意外终止了正在生成的字幕进程，您也可以恢复进度。
5. **可配置**：KonnyakuGPT即使处理具有不寻常设置和包含视频特定专业术语（例如角色名称）的视频也能很好地工作，这在日本动画中很常见。
   只要输入描述视频背景并包含可能出现的特殊术语的背景信息，KonnyakuGPT就能推断情节并创建有意义且正确处理特殊术语的字幕。

以下是一张来自日本动画《大耳鼠芝比》的剧集截图，配有KonnyakuGPT生成的翻译字幕：

![example](../res/example.png)

KonnyakuGPT 不仅正确地处理了并翻译了角色的名字（日语："エリ" / 英语："Eri" / 中文："惠理"），还识别出了场景中的一个特殊物体，并推断出了其功能（场景中有一把"幻想枪"，可以发射特殊的光线，注入特定的思想到生物体中，效果持续24小时）。

## 示例用法

要使用预定义的方法创建字幕，只需导入默认字幕生成器，输入背景信息，并调用生成器。

**KonnyakuGPT使用OpenAI的API服务；在调用字幕生成器之前，您必须配置API密钥。**
最简单的方法是在运行生成字幕的Python脚本之前设置环境变量"OPEN_AI_API_KEY"。
例如：

```bash
# 如果您使用类Unix操作系统，请运行此行
export OPENAI_API_KEY=<your-api-key>
# 如果您使用Windows，请运行此行
set OPENAI_API_KEY=<your-api-key>

python <调用KonnyakuGPT的脚本>
```

以下是生成藤子·F·不二雄的作品《大耳鼠芝比》中一集动画字幕的示例脚本：

```Python
from pathlib import Path

from src.subtitle_generation import DefaultGenerator

generator = DefaultGenerator(quality_preset='medium')


generator.generate_subtitles(video_path=Path('/home/trent/Downloads/episode.mp4'),
                            output_path=Path('test.srt'),
                            video_background=\
"""该视频由日本动画《大耳鼠芝比》的两集组成。
第一集的标题是"ワンダユウは占い師？"，第二集的标题是"エリさまは美少女"。
在每一集的开头都有一个介绍屏幕，其中大声宣布了该集的标题。

《大耳鼠芝比》中的一些特殊术语：

A. 角色：

    1. 春日惠理（Kasuga Eri），经常被称为"Eri"（中文：惠理）、"Eri-chan"、"Eri-sama"（中文：惠理大人）和"Kasuga"（中文：春日）。她是一个阳刚的女孩，喜欢她的同学内木，根本不想嫁给来自马尔星的王子。
    2. 芝比（Chimpui，中文：芝比）。拥有超能力的外星老鼠。
    3. 旺达（Wanderyu，中文：旺达）。拥有超能力的外星狗。
    4. 内木翔（Shou Uchiki），经常被称为"Uchiki"（中文：内木）。是一名优秀的学生。惠理喜欢他。
    5. 小金山诗奈美（Sunemi Koganeyama），经常被称为"Sunemi" / "Sunemi-chan" / "Sunemi-san"（中文：诗奈美）。一个喜欢炫耀事物的富家女孩，尤其是她家族的财富。
    6. 大江山政男（Masao Oeyama），经常被称为"Oeyama"（中文：大江山）。一名强壮的六年级生，有时欺负同学，尤其是Uchiki。
    7. 木常小政（Shosei Kitsune），经常被称为"Shosei"（中文：小政）或"Kitsune"（中文：木常）。Oeyama的跟班。
    8. 藤野小莹（Hotaru Fujino），经常被称为"Hotaru"（中文：小莹）。一个爱幻想的女孩，经常梦想着幻想的事情，比如一个帅气的王子来娶她。
    9. 吕诺夫（Lulealv），经常被称为"Lulealv Denka"（中文：吕诺夫殿下）或简称"Denka"（殿下）。来自马尔星的王子，希望娶惠理。然而，惠理根本不想嫁给他。

    Chinpui和Wanda都来自马尔星。
    这颗星球在英语中也被称为Mahl星球，
    或中文为"玛尔星"。

B. 其他：
    1. Mahl（马尔），Chinpui和Wanda来自的星球的名字。
    2. 科法（Kahou / かほう）。"科法"是来自马尔星的特殊、先进且方便使用的技术。
    Chinpui和Wanda都使用科法。
""",
                            target_language='简体中文')
```

## 致敬

本项目是为了纪念著名的日本漫画家藤本弘（Hiroshi Fujimoto），也被称为藤子·F·不二雄（Fujiko F. Fujio）而开发的。
"KonnyakoGPT"的名称灵感来自于他最著名的作品《哆啦A梦》中的"ほんやくコンニャク"（"翻译蒟蒻"）。
