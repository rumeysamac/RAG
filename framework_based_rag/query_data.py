import argparse
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
  
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Aşağıdaki bağlamı kullanarak soruyu yanıtla:

{context}

---

Soru: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

  
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

   
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    if len(results) == 0 or results[0][1] < 0.7:
        print("Eşleşen uygun bir belge bulunamadı.")
        return

    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(f"\n--- OLUŞTURULAN PROMPT ---\n{prompt}")

   
    model = ChatOpenAI(model="gpt-4o-mini")
    response_text = model.invoke(prompt)

    formatted_response = f"Yanıt: {response_text.content}"
    print(f"\n--- YANIT ---\n{formatted_response}")

if __name__ == "__main__":
    main()