"""Automatic speech recognition model service

Backend: whisper-small (local, cuda if available else cpu)
"""
from pathlib import Path

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa


class Transcriber:
    
    def __init__(self):
        # load model and processor
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="japanese", task="transcribe")
    
    def __call__(self, audio_path: Path):
        array, sampling_rate = librosa.load(audio_path, sr=16000)
        sample = {"path": audio_path, "array": array, "sampling_rate": sampling_rate}

        input_speech = sample
        input_features = self.processor(input_speech["array"], sampling_rate=input_speech["sampling_rate"], return_tensors="pt").input_features

        # generate token ids
        predicted_ids = self.model.generate(input_features, forced_decoder_ids=self.forced_decoder_ids, return_timestamps=True)
        # decode token ids to text
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True, return_timestampes=True)
        
        return transcription
    