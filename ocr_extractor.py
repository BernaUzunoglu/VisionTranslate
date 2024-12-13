import pytesseract
from PIL import Image
import cv2
import numpy as np
from transformers import MarianMTModel, MarianTokenizer

# Tesseract'ın sistemdeki yolunu belirt
# Örneğin, Windows için: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Eğer Linux/Mac kullanıyorsanız, genelde ek bir ayar gerekmez.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Resmi yükle ve işleme adımları
def process_image(image_path):
    # Resmi Pillow ile aç
    image = Image.open(image_path)

    # Görüntüyü opencv formatına çevir
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Gri tonlama (opsiyonel, OCR'ı iyileştirebilir)
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # OCR ile metni al
    text = pytesseract.image_to_string(gray_image, lang='eng')

    return text


# Çeviri işlemi - Hugging Face'in Helsinki-NLP modelini kullanarak
def translate_text(text, target_lang='tr'):
    # MarianMTModel ve MarianTokenizer'ı yükle
    model_name = f'Helsinki-NLP/opus-mt-en-{target_lang}'  # 'en' (İngilizce) -> 'tr' (Türkçe) çeviri modeli
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # Metni çeviri için tokenize et
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True, truncation=True))

    # Çevrilen metni çözümle
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Örnek bir resim dosyasını işleme
if __name__ == "__main__":
    image_path = "Truthfulness.png"  # Resim dosyasının yolu
    extracted_text = process_image(image_path)

    print("Çıkarılan Metin:")
    print(extracted_text)

    translated_text = translate_text(extracted_text)
    print("Çevrilen Metin:")
    print(translated_text)