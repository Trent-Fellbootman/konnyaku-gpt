from image_to_text import ImageToTextModelService
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration






class BlipLarge(ImageToTextModelService):

    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

    def call(self, image: Image) -> str:
        text = "a photography of"
        inputs = self.processor(image, text, return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)