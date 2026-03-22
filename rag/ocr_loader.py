import fitz
import pytesseract
from PIL import Image
import io
from langchain_core.documents import Document

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def ocr_pdf(file_path):
    doc = fitz.open(file_path)
    documents = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img)
        if text.strip():
            documents.append(Document(
                page_content=text,
                metadata={"source": file_path, "page": i + 1}
            ))
    return documents