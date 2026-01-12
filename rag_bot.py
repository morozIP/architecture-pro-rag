import os
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# 1. API токен
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 2. Эмбеддинги и база
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'token': hf_token}
)
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# 3. Настройка модели через Chat-интерфейс
# Это решит проблему с "Supported task: conversational"
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="conversational", # Меняем задачу на ту, которую требует провайдер
    max_new_tokens=250, # Ограничим длину, чтобы не частил
    temperature=0.01,   # Минимальная температура для стабильности
    repetition_penalty=1.2, # ЗАЩИТА ОТ ЗАЦИКЛИВАНИЯ
)
chat_model = ChatHuggingFace(llm=llm)

# 4. Промпт в формате чата (System + User)
prompt = ChatPromptTemplate.from_messages([
    ("system", """Ты — интеллектуальный помощник QuantumForge Software. 
Отвечай строго на РУССКОМ языке. Используй только контекст. 
Если ответа нет, пиши: "Я не знаю".

ТЕХНИКА FEW-SHOT:
Вопрос: Кто управляет Void Legion?
Рассуждение: В документах указано, что Zarn Kael — их чемпион.
Ответ: Легионом управляет Zarn Kael.

ТЕХНИКА CHAIN-OF-THOUGHT:
Сначала пиши ход мыслей (Рассуждение), затем Ответ."""),
    ("user", "КОНТЕКСТ: {context}\n\nВОПРОС: {question}\n\nРАССУЖДЕНИЕ (на русском):")
])

# 5. Цепочка
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | chat_model
    | StrOutputParser()
)

if __name__ == "__main__":
    print("\n--- Бот запущен! ---")
    while True:
        query = input("\nСотрудник: ")
        if query.lower() in ["выход", "exit"]: break
        print("\nБот думает...")
        try:
            print(f"\nОтвет:\n{rag_chain.invoke(query)}")
        except Exception as e:
            print(f"Ошибка: {e}")