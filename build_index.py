import os
import time  # Добавляем модуль для замера времени
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Указываем путь к твоей папке с данными
DATA_PATH = "knowledge_base/"


def create_vector_db():
    start_total = time.time()  # Засекаем общее время

    print("--- Загрузка документов... ---")
    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"Загружено документов: {len(documents)}")

    # 2. Нарезаем текст на кусочки (чанки)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"Текст разбит на {len(docs)} чанков.")

    # 3. Инициализируем модель эмбеддингов
    print("--- Загрузка модели эмбеддингов... ---")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 4. Создаем векторную базу данных (замеряем именно этот этап)
    print("--- Генерация эмбеддингов и создание индекса FAISS... ---")
    start_indexing = time.time()

    vector_db = FAISS.from_documents(docs, embeddings)

    end_indexing = time.time()
    indexing_time = end_indexing - start_indexing

    # 5. Сохраняем базу
    vector_db.save_local("faiss_index")

    end_total = time.time()
    total_time = end_total - start_total

    print("\n" + "=" * 30)
    print(f"ИТОГИ ЗАМЕРОВ:")
    print(f"Количество чанков: {len(docs)}")
    print(f"Время индексации: {indexing_time:.2f} сек.")
    print(f"Общее время выполнения: {total_time:.2f} сек.")
    print("=" * 30)
    print("Индекс сохранен в папку 'faiss_index'")


if __name__ == "__main__":
    create_vector_db()