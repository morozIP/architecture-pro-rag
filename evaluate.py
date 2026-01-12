import json
import time
from tg_bot import rag_chain  # Импортируйте вашу цепочку


def run_evaluation():
    with open("golden_questions.json", "r", encoding="utf-8") as f:
        test_set = json.load(f)

    results = []
    for item in test_set:
        query = item['question']
        print(f"Тестируем: {query}")

        start = time.time()
        response = rag_chain.invoke(query)
        latency = time.time() - start

        # Простая проверка: если в ответе есть маркеры, значит информация найдена
        success = "$%" in response or "Я не знаю" in response

        results.append({
            "query": query,
            "response": response,
            "latency": round(latency, 2),
            "status": "PASS" if success else "FAIL"
        })

    with open("logs.jsonl", "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("Аналитика завершена. Лог сохранен в logs.jsonl")


if __name__ == "__main__":
    run_evaluation()