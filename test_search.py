from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Загружаем ту же модель
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Загружаем созданную базу
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Твой тестовый запрос (используй свои вымышленные термины!)
query = "Кто такой Zarn Kael?" # Замени на свой вопрос

# Ищем 3 самых похожих отрывка
results = db.similarity_search(query, k=3)

print(f"\nРезультаты поиска по запросу: '{query}'\n")
for i, res in enumerate(results):
    print(f"Отрывок №{i+1}:")
    print(res.page_content)
    print("-" * 30)