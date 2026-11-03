def self_rag_evaluator(context: str, answer: str) -> str:
  context_words = set(context.lower().split())
  answer_words = set(answer.lower().split())

  overlap = len(context_words.intersection(answer_words))
  score = overlap / max(len(answer_words), 1)

  if score < 0.2:
    return "❌ REDDEDİLDİ: Yanıt bağlamla uyuşmuyor!"
  else:
    return f"✅ ONAYLANDI: Uyum Skoru: {score:.2f}"

orijinal_metin = "Şirketimiz 2025 yılında %20 büyüme kaydetmiştir."
uretilen_yanıt = "Şirketimiz geçen yıl yüzde 20 oranında büyüdü."

sonuc = self_rag_evaluator(orijinal_metin, uretilen_yanıt)
print("Self-RAG Değerlendirme Sonucu:", sonuc)

orijinal_metin = "Şirketimiz 2025 yılında %20 büyüme kaydetmiştir."
uretilen_yanıt = "Şirketimiz 2025 yılında %20 büyüme kaydetmiştir."

sonuc = self_rag_evaluator(orijinal_metin, uretilen_yanıt)
print("Self-RAG Değerlendirme Sonucu:", sonuc)