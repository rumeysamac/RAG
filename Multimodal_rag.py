import base64
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode("utf-8")


def openai_multimodal_rag(image_path: str, user_question: str):
  try:
    base64_image = encode_image(image_path)
  except FileNotFoundError:
    return (
        "Önce test için uygun bir görsel dosyası (örneğin satis_grafigi.png)"
        " eklemelisin!"
    )

  response = client.chat.completions.create(
      model="gpt-4o", 
      messages=[{
          "role": "user",
          "content": [
              {
                  "type": "text",
                  "text": (
                      f"Aşağıdaki görsele bakarak soruyu yanıtla:"
                      f" {user_question}"
                  ),
              },
              {
                  "type": "image_url",
                  "image_url": {
                      "url": f"data:image/jpeg;base64,{base64_image}"
                  },
              },
          ],
      }],
      max_tokens=300,
  )

  return response.choices[0].message.content


soru = "Grafiğe göre en yüksek satış hangi bölgede gerçekleşmiş?"
sonuc = openai_multimodal_rag("satis_grafigi.png", soru)
print(sonuc)