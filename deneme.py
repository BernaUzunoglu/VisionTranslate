import pytesseract
from PIL import Image
import cv2
from transformers import MarianMTModel, MarianTokenizer

# Model ve Tokenizer yükleme
model_name = "Helsinki-NLP/opus-tatoeba-en-tr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


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

# Çeviri fonksiyonu
def translate(text, tokenizer, model):
    # Girdi metni için token oluşturma
    inputs = tokenizer.encode(text, return_tensors="pt", padding=True)
    # Model ile çeviri
    translated_tokens = model.generate(inputs)
    # Çeviriyi çözümleme
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text


image_path = "Truthfulness.png"  # Resim dosyasının yolu
extracted_text = process_image(image_path)

print("Çıkarılan Metin:")
print(extracted_text)

translated_text = translate(extracted_text,tokenizer, model)
print("Çevrilen Metin:")
print(translated_text)

