import os
import time
import logging
from datetime import datetime
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Настройка логирования в файл
logging.basicConfig(
    filename='update_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

DATA_PATH = "knowledge_base/"
INDEX_PATH = "faiss_index"


def update_vector_db():
    start_time = time.time()
    try:
        # 1. Загрузка
        loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        documents = loader.load()

        # 2. Разбиение
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # 3. Эмбеддинги
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        # 4. Обновление индекса
        vector_db = FAISS.from_documents(docs, embeddings)
        vector_db.save_local(INDEX_PATH)

        duration = time.time() - start_time
        log_msg = f"УСПЕХ: Индекс обновлен. Файлов: {len(documents)}, Чанков: {len(docs)}, Время: {duration:.2f}с"
        print(log_msg)
        logging.info(log_msg)

    except Exception as e:
        err_msg = f"ОШИБКА при обновлении индекса: {e}"
        print(err_msg)
        logging.error(err_msg)


if __name__ == "__main__":
    update_vector_db()