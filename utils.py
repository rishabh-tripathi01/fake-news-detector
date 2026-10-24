import re

def clean_text(text):
    """Dataset ki 'cheating' wali cheezein hatata hai (Reuters tag, links)
    taaki model sirf asli language patterns seekhe."""
    text = str(text)
    text = re.sub(r"^[^()]{0,60}\(Reuters\)\s*[-–—]?\s*", "", text)  # 'WASHINGTON (Reuters) -'
    text = re.sub(r"(?i)reuters", "", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)                      # links
    text = re.sub(r"(?i)pic\.twitter\.com/\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
