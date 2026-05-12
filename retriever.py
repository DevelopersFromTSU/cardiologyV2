import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

# Подключаемся к локальной базе Qdrant
qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "medical_docs"

# Меняем модель
print("⏳ Загрузка мощной модели для поиска...")
model = SentenceTransformer('intfloat/multilingual-e5-base')
model.max_seq_length = 512
print("✅ Модель готова!")


def get_relevant_chunks(query, top_k=3):
    # <-- НОВЫЕ СТРОЧКИ: Добавляем префикс query: для вопроса
    query_vector = model.encode("query: " + query).tolist()

    # <-- НОВЫЕ СТРОЧКИ: Используем актуальный API Qdrant
    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )
    search_result = response.points

    retrieved_texts = []
    print(f"\n🔍 Вопрос: '{query}'\n" + "=" * 50)

    # 3. Распаковываем результаты
    for i, hit in enumerate(search_result, 1):
        text = hit.payload.get('text', '')
        page = hit.payload.get('page', 'Неизвестно')
        retrieved_texts.append({"page": page, "text": text})

        print(f"[Совпадение {i} | Страница: {page} | Точность: {hit.score:.4f}]")
        print(f"{text[:300]}...\n{'-' * 50}")

    return retrieved_texts

if __name__ == "__main__":
    # Тестовый запрос (как раз по теме твоих страниц)
    test_query = "Что нам говорит значение больше или равно 140 у  систолического артериального давления?"
    chunks = get_relevant_chunks(test_query, top_k=3)