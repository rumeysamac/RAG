import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

def uzun_dokuman_analizi(dosya_yolu, soru):
  try:
  
    with open(dosya_yolu, "r", encoding="utf-8") as f:
      tum_metin = f.read()

    print(
        f"📄 '{dosya_yolu}' başarıyla okundu. Modelin devasa bağlam penceresine"
        " gönderiliyor..."
    )

    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[
            {
                "role": "system",
                "content": (
                    "Sen uzman bir doküman analiz asistanısın. Sana sunulan"
                    " devasa metni eksiksiz incele ve kullanıcının sorusuna"
                    " metindeki bilgilere sadık kalarak detaylı ve net bir yanıt"
                    " ver."
                ),
            },
            {"role": "user", "content": f"Doküman İçeriği:\n\n{tum_metin}"},
            {"role": "user", "content": f"Soru: {soru}"},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content

  except FileNotFoundError:
    return (
        "❌ Hata: Belirtilen dosya bulunamadı. Lütfen dosya yolunu kontrol et."
    )
  except Exception as e:
    return f"❌ Bir hata oluştu: {e}"


if __name__ == "__main__":
  
  hedef_dosya = "buyuk_rapor.txt"
  kullanici_sorusu = (
      "Bu raporda geçen ana stratejik hedefler ve riskler nelerdir?"
  )

  sonuc = uzun_dokuman_analizi(hedef_dosya, kullanici_sorusu)
  print("\n--- Uzun Bağlam RAG Yanıtı ---")
  print(sonuc)