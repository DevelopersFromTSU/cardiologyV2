import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import re

# <-- НОВЫЙ ИМПОРТ ДЛЯ ЛОКАЛЬНОЙ МОДЕЛИ
from sentence_transformers import SentenceTransformer

load_dotenv()

# Подключаемся к Qdrant (через Docker)
qdrant = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "medical_docs"

print("Загрузка медицинской модели RuBioRoBERTa... (при первом запуске может занять время)")
# <-- ЗАГРУЖАЕМ ЛОКАЛЬНУЮ МОДЕЛЬ
# Используем популярную версию RuBioRoBERTa с Hugging Face
# Используем популярную версию RuBioRoBERTa с Hugging Face
model = SentenceTransformer('alexyalunin/RuBioRoBERTa')
model.max_seq_length = 512  # <-- НОВАЯ СТРОЧКА: жесткий лимит
print("Модель успешно загружена!")


def init_qdrant(vector_size=1024):
    """Создает коллекцию, если ее еще нет."""
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            # Векторы RoBERTa обычно имеют размер 768, как и у Gemini
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def get_embedding(text):
    """Превращает текст в вектор с помощью локальной модели RuBioRoBERTa."""
    # нейросеть читает текст и выдает вектор.
    # tolist() превращает его в обычный список Python, который понимает Qdrant
    vector = model.encode(text).tolist()
    return vector


def get_smart_chunks(text, chunk_size=1500, chunk_overlap=250):
    # Теперь мы режем ТОЛЬКО по главной метке страницы
    headers_to_split_on = [
        ("#", "Header 1"), # Это будет наша "Страница X"
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_splits = markdown_splitter.split_text(text)

    # Теперь RecursiveCharacterTextSplitter будет нарезать эти большие куски
    # страниц на блоки ПОЧТИ ВЫРАВНЕННЫЕ по 1500 символов.
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    return char_splitter.split_documents(md_splits)


def process_and_upload_to_qdrant(refined_data, page_num):
    init_qdrant()
    text = refined_data.get("refined_text", "")
    if not text:
        return

    chunks = get_smart_chunks(text)
    points = []

    for i, chunk_doc in enumerate(chunks):
        chunk_text = chunk_doc.page_content
        chunk_headers = chunk_doc.metadata

        # --- НОВАЯ ЛОГИКА ИЗВЛЕЧЕНИЯ СТРАНИЦЫ ---
        current_page = page_num  # По умолчанию берем то, что пришло (0)

        # Ищем номер страницы в заголовках (Header 1 или Header 2)
        for header_value in chunk_headers.values():
            if "Страница" in header_value:
                # Вытаскиваем только цифры из строки "Страница 188"
                match = re.search(r'\d+', header_value)
                if match:
                    current_page = int(match.group())
                    break
        # ----------------------------------------

        vector = get_embedding(chunk_text)
        point_id = str(uuid.uuid4())

        payload = {
            "text": chunk_text,
            "page": current_page,  # Теперь здесь будет реальный номер страницы
            "chunk_index": i
        }
        payload.update(chunk_headers)

        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"✅ Успешно загружено {len(points)} чанков с корректными номерами страниц.")