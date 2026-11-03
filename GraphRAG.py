
knowledge_graph = {
    "RAG": {
        "baglantilar": {
            "kullanir": ["Vektör Veritabanı", "Büyük Dil Modeli"],
            "engeller": ["Halüsinasyon"],
        }
    },
    "Vektör Veritabanı": {
        "baglantilar": {"saklar": ["Metin Parçaları (Chunk)", "Embedding"]}
    },
}


def query_graph(entity: str, relation: str):
  if (
      entity in knowledge_graph
      and relation in knowledge_graph[entity]["baglantilar"]
  ):
    return knowledge_graph[entity]["baglantilar"][relation]
  return "Bu varlık veya ilişki grafikte bulunamadı."


print("Soru 1: RAG sistemi neleri kullanır?")
sonuc_1 = query_graph("RAG", "kullanir")
print(f"Çıktı: {sonuc_1}\n")

print("Soru 2: RAG sistemi neyi engeller?")
sonuc_2 = query_graph("RAG", "engeller")
print(f"Çıktı: {sonuc_2}")