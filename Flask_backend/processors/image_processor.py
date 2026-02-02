import pytesseract
from PIL import Image
from io import BytesIO

def extract_text_from_image(file):
    image=Image.open(BytesIO(file.read()))
    return pytesseract.image_to_string(image)