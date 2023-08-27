from .transcriber import TranscriberModelService
from pathlib import Path

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa


class Whisper(TranscriberModelService):
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # load model and processor
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(self.device)
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="japanese", task="transcribe")
    
    # override
    def call(self, audio_path: Path):
        array, sampling_rate = librosa.load(audio_path, sr=16000)
        sample = {"path": audio_path, "array": array, "sampling_rate": sampling_rate}

        input_speech = sample
        input_features = self.processor(input_speech["array"], sampling_rate=input_speech["sampling_rate"], return_tensors="pt").input_features.to(self.device)

        # generate token ids
        predicted_ids = self.model.generate(input_features, forced_decoder_ids=self.forced_decoder_ids, return_timestamps=True)
        # decode token ids to text
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True, return_timestampes=True)
        
        return transcription
    
    def get_description() -> str:
        return \
"""
This model named "Whisper" is an ASR model prompt-tuned for Japanese transcribing.
The model should not be assumed to have world knowledge, and its outputs are likely to be very inaccurate.
Problems in its outputs may include but not limited to:

1. Incorrect recognition of kanas, e.g., recognizing "ぶ" as "ぷ".
2. Failure to identify prolonged kanas, e.g., recognizing "えい" as "え".
3. Incorrect semantics. These include recognizing hiraganas as katakanas and vice versa,
translating kanas to wrong Kanjis, incorrect semantic grouping & punctuation (e.g., "はい、そうです。" v.s. "配送です。"), etc.
"""
