from .image_to_text import ImageToTextModelService
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


class BlipLarge(ImageToTextModelService):

    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(self.device)

    # override
    def call(self, image: Image) -> str:
        text = ''
        inputs = self.processor(image, text, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)
    
    # override
    def get_description() -> str:
        return \
"""
This model named "blip-image-captioning-large" is able to generate captions for images.
However, its output is likely to be very inaccurate.
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
"""