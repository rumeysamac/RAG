
user_role = "guest"

database_chunks = [
    {"chunk": "Genel şirket politikaları ve çalışma saatleri.", "role": "guest"},
    {"chunk": "2026 Q3 Mali Bütçe ve Maaş Tabloları.", "role": "admin"},
]


def secure_retrieve(role: str):
  
  allowed_chunks = [
      item["chunk"] for item in database_chunks if item["role"] == role
  ]
  return allowed_chunks

sonuc = secure_retrieve(user_role)
print(f"Kullanıcı Rolü ({user_role}) İçin Çekilen Bağlam:")
print(sonuc)