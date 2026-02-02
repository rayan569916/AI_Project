from PyPDF2 import PdfReader
from io import BytesIO

def extract_text_from_pdf_file(file):
    pdf=PdfReader(BytesIO(file.read()))
    text=""
    for page in pdf:
        if page.extract_text():
            text+=page.extract_text()
    return text