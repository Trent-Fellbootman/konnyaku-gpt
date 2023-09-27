# 魔芋GPT: 多模态GPT实现更精准的动画字幕生成

**注：中文README使用ChatGPT翻译**

[English](../README.md) [日本語](./ja.md)

魔芋GPT是一款AI驱动的高质量且易于使用的字幕生成器，主要用于日本动画。

## 主要特点

1. **易于使用**：只需一次函数调用，即可从视频生成高质量的字幕文件。
2. **高质量**（取决于您的预算）：使用“中等”质量预设，每20分钟的视频约需花费4美元，魔芋GPT能够正确处理80%的视频专用术语（角色名称等），并生成构建逻辑连贯的字幕。**由魔芋GPT生成的字幕可让没有日语知识的观众观看日本动画并理解情节，无需猜测发生了什么。**
3. **翻译支持**：魔芋GPT可以为任何语言创建翻译字幕。
4. **暂停和恢复**：生成字幕时会定期保存进度。即使意外终止正在生成字幕的进程，您也可以恢复进度。
5. **可配置**：即使您的视频具有不寻常的设置并包含视频特定的专业术语（例如角色名称），这在日本动画中很常见，魔芋GPT也能够正常工作。只要输入描述视频背景并包括可能出现的特殊术语的背景信息，魔芋GPT将推断情节并创建有意义的字幕，并正确处理特殊术语。

以下是从日本动画《大耳鼠芝比》中截取的一集的屏幕截图，其中包括由魔芋GPT生成的翻译字幕：

![示例](../res/example.png)

魔芋GPT不仅正确识别和翻译了角色的名称（日语：“エリ” / 英语：“Eri” / 中文：“惠理”），还确定了场景中的特殊对象并推断其功能（场景中有一把“想象枪”，可以发射特殊的射线，注入特定的思想到生物中，效果持续24小时）。

## 安装

可以通过pip安装魔芋GPT：

```bash
pip install konnyaku-gpt
```

## 配置和使用

### Open AI设置

**魔芋GPT使用Open AI的API服务；在调用字幕生成器之前，您必须配置API密钥。**最简单的方法是在运行生成字幕的Python脚本之前设置环境变量“OPENAI_API_KEY”。例如：

```bash
# 如果您使用类Unix的操作系统，请运行此行
export OPENAI_API_KEY=<your-api-key>
# 如果您使用Windows，请运行此行
set OPENAI_API_KEY=<your-api-key>

python <调用魔芋GPT的脚本>
```

### 使用方法

要使用预定义方法创建字幕，只需导入默认字幕生成器，输入背景信息，并调用生成器。

以下是为藤子·F·不二雄（Fujiko F. Fujio）的动画《大耳鼠芝比》的一集生成字幕的示例脚本：

```Python
from pathlib import Path

from konnyaku_gpt.subtitle_generation import DefaultGenerator
from konnyaku_gpt.tricks import simple_split_subtitle_file

# 建议使用“medium”预设以在质量和成本之间取得平衡
generator = DefaultGenerator(quality_preset='medium')

output_path = Path('/output/srt/path')

# 使用AI生成字幕！
generator.generate_subtitles(video_path=Path('/path/to/Chimpui/episode/mp4'),
                             output_path=output_path,
                             video_background=\
"""该视频包括来自日本动画《大耳鼠芝比》的两集。
第一集的标题是“レッツゴー銀河レース”；第二集的标题是“はじめまして、ルルロフです”。
每集开始时都有一个介绍画面，其中宣布该集的标题。

《Chimpui》中的一些特殊术语：

A. 角色：

    1. 春日エリ（Kasuga Eri），通常称为“Eri”（中文：惠理）、“Eri-chan”、“Eri-sama”（中文：惠理大人）和“Kasuga”（中文：春日）。她是一个男孩气的女孩。她喜欢她的同学内木（Uchiki）并且不想嫁给Mahl星球的王子。
    2. Chimpui（チンプイ，中文：芝比）。一个拥有超能力的外星老鼠。
    3. Wanderyu（ワンダユウ，中文：旺达）。一个拥有超能力的外星狗。
    4. 内木翔（Shou Uchiki），通常称为“Uchiki”（中文：内木）。一个优秀的学生。Eri喜欢他。
    5. 小金山スネ美（Sunemi Koganeyama），通常称为“Sunemi” / “Sunemi-chan” / “Sunemi-san”（中文：诗奈美）。一个喜欢炫耀的富家女孩，尤其是她家族的财富。
    6. 大江山政男（Masao O

eyama），通常称为“Oeyama”（中文：大江山）。一个强壮的六年级生，有时会欺负同学，尤其是Uchiki。
    7. 木常小政（Shosei Kitsune），通常称为“Shosei”（中文：小政）或“Kitsune”（中文：木常）。Oeyama的助手。
    8. 藤野ほたる（Hotaru Fujino），通常称为“Hotaru”（中文：小莹）。一个梦想着幼稚而不切实际的事情，比如帅气的王子来娶她的女孩。
    9. Lulealv，通常称为“Lulealv Denka”（中文：吕诺夫殿下）或简称“Denka”（殿下）。Mahl星球的王子，希望娶Eri。然而，Eri根本不想嫁给他。

    Chinpui和Wanda都来自Mahl星球。
    这个星球在英语中也称为Mahl星球，中文称为“玛尔星”。

B. 其他：
    1. Mahl（マール），Chinpui和Wanda来自的星球的名称。
    2. Kahou（科法 / かほう）。"科法"是来自Mahl星球的特殊、先进且方便使用的技术。Chinpui和Wanda都使用科法。
""",
                             target_language='简体中文',
                             workspace_path=Path('final-episode-workspace'))

# 拆分字幕，使每个字幕元素都很短
simple_split_subtitle_file(output_path)

```

## 故障排除

### 中国用户特别注意

由于中国境内某些互联网访问存在已知问题，您可能需要使用代理服务器来使用Open AI服务。为魔芋GPT设置代理的方法很简单，只需在运行魔芋GPT的控制台中通过环境变量指定代理服务器地址和端口，例如：

```bash
# 如果您使用类Unix的操作系统，请运行此行
export ALL_PROXY=127.0.0.1:<your-proxy-port>
# 如果您使用Windows，请运行此行
set ALL_PROXY=127.0.0.1:<your-proxy-port>
```

### CUDA设置

魔芋GPT在生成字幕时会调用本地部署的深度学习模型。默认情况下，如果您的计算机配备了支持CUDA的显卡，它会尝试将模型放在GPU上运行。**但是，如果您的GPU VRAM少于8GB，则可能无法容纳模型，并可能引发错误。**

在这种情况下，您需要告诉魔芋GPT不要使用CUDA，最简单的方法是通过环境变量将GPU屏蔽，即：

```bash
# 如果您使用类Unix的操作系统，请运行此行
export CUDA_VISIBLE_DEVICES=''
# 如果您使用Windows，请运行此行
set CUDA_VISIBLE_DEVICES=''
```

## 致敬

本项目是为了纪念著名的日本漫画家藤本弘（Fujiko F. Fujio）而开发的。项目名称“KonnyakoGPT”受到了他最受欢迎的作品《哆啦A梦》中的“ほんやくコンニャク”（翻译魔芋）的启发。
