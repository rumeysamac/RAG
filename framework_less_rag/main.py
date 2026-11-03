import sys
import os
import numpy as np

from rag import LightweightRAG

def main():
    
    documents = [
        "Python programlama dili, veri bilimi ve yapay zeka projelerinde yaygın olarak kullanılır.",
        "Makine öğrenmesi algoritmaları verilerden öğrenerek tahminler yapar.",
        "Kanser hücrelerinin tespiti için derin öğrenme modelleri geliştirilmektedir.",
        "Evde lezzetli bir menemen yapmak için domates sosu, biber ve yumurta gereklidir.",
        "Türk mutfağının vazgeçilmez lezzetlerinden biri de zeytinyağlı yaprak sarma'dır.",
        "Beşiktaş, Süper Lig'in köklü ve tarihi futbol kulüplerinden biridir.",
        "Futbol maçlarında taktiksel dizilişler ve orta saha kontrolü çok önemlidir."
    ]

   
    rag = LightweightRAG(model_name="all-MiniLM-L6-v2", n_clusters=2)

    
    rag.index(documents, adj_threshold=0.50)

    
    print("\n---  STANDART ARAMA TESTİ ---")
    query = "Yapay zeka ve python projeleri"
    results = rag.search(query, top_k=2, use_ivf=True)
    
    for doc, score in results:
        print(f"Skor: {score:.4f} | İçerik: {doc}")

    
    print("\n--- 🕸️ GRAPH TABANLI ARAMA TESTİ ---")
    graph_results = rag.graph_search(query, top_k=2, hops=1)
    
    for doc, score in graph_results:
        print(f"Skor: {score:.4f} | İçerik: {doc}")

     
    print("\n--- 📊 GÖRSELLEŞTİRME ÇALIŞTIRILIYOR ---")
    rag.visualise(query=query)

if __name__ == "__main__":
    main()