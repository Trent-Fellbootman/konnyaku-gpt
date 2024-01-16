from pathlib import Path
from .transcriber import TranscriberModelService
from openai import OpenAI
from moviepy.audio.io.AudioFileClip import AudioFileClip
import logging


class WhisperCloud(TranscriberModelService):
    
    def __init__(self) -> None:
        self.client = OpenAI()
        
    def call(self, audio_path: Path) -> str:
        if AudioFileClip(audio_path).duration < 0.2:
            # return empty string if audio is too short
            return ''
        
        with open(audio_path, 'rb') as f:
            response = self.client.audio.transcriptions.create(
                model='whisper-1',
                file=f,
            )
        
        return response.text
    
    @staticmethod
    def get_description() -> str:
        return \
"""
This model named "Whisper" is an ASR model for transcription.
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
"""
