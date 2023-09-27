# pip install accelerate
import torch
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from image_to_text import ImageToTextModelService


class Blip2Opt(ImageToTextModelService):
    def __init__(self):
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="auto")

    def call(self, image: Image) -> str:
        question = "What does the picture say?"
        inputs = self.processor(image, question, return_tensors="pt").to("cuda", torch.float16)

        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)




