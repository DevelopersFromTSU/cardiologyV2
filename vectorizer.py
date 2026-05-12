import os
import json
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from sentence_transformers import SentenceTransformer
import re

load_dotenv()

JSON_FOLDER = "./result"
COLLECTION_NAME = "medical_docs"

# Подключаемся к Qdrant (через Docker)
qdrant = QdrantClient(url="http://localhost:6333")

print("⏳ Загрузка мощной модели для эмбеддингов...")
model = SentenceTransformer('intfloat/multilingual-e5-base')
model.max_seq_length = 512
print("Модель успешно загружена!")


def clean_excessive_whitespace(text):
    if not text:
        return text
    # <-- НОВЫЕ СТРОЧКИ: Удаляем библиографические ссылки
    # Ищет квадратные скобки, внутри которых только цифры, запятые, пробелы или тире
    text = re.sub(r'\[\d+[\d\s,\-]*\]', '', text)
    # 1. Заменяем 3 и более переносов строк на стандартные 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 2. Убираем пробелы и табуляцию в конце строк
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    # 3. Заменяем 2+ пробела между словами на один
    text = re.sub(r'(?<=\S)[ \t]{2,}', ' ', text)
    # <-- НОВЫЕ СТРОЧКИ: Убираем "висячие" пробелы перед знаками препинания
    # Если ссылка была перед точкой (например "пациента [2]."), останется пробел перед точкой
    text = re.sub(r' \.', '.', text)
    text = re.sub(r' \,', ',', text)

    return text


def merge_small_chunks(chunks, min_size=300):
    """
    Проходит по чанкам и приклеивает слишком маленькие к предыдущим.
    """
    if not chunks:
        return []

    merged_chunks = []
    for chunk in chunks:
        # Если список пуст, просто добавляем первый чанк
        if not merged_chunks:
            merged_chunks.append(chunk)
            continue

        # Если текущий чанк меньше минимального размера (например, 300 символов)
        if len(chunk.page_content) < min_size:
            # Приклеиваем его текст к тексту предыдущего чанка
            merged_chunks[-1].page_content += "\n\n" + chunk.page_content
            # Обновляем метаданные (добавляем новые заголовки, если они появились)
            merged_chunks[-1].metadata.update(chunk.metadata)
        else:
            # Если чанк нормального размера, оставляем как есть
            merged_chunks.append(chunk)

    return merged_chunks

def init_qdrant(vector_size=768):
    """Создает коллекцию, если ее еще нет."""
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )


def get_embedding(text):
    return model.encode(text).tolist()


def get_smart_chunks(text, chunk_size=1200, chunk_overlap=300):
    # 1. Сначала бьем по Markdown заголовкам
    headers_to_split_on = [
        ("#", "Header 1"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_splits = markdown_splitter.split_text(text)

    # 2. Затем режем длинные абзацы по символам
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True
    )

    return char_splitter.split_documents(md_splits)


def process_and_upload(text_to_upload, page_keywords, page_num):
    init_qdrant()
    if not text_to_upload:
        return

    text_to_upload = clean_excessive_whitespace(text_to_upload)
    # 1. Режем текст (получаем сырые чанки)
    raw_chunks = get_smart_chunks(text_to_upload, chunk_size=1200, chunk_overlap=300)
    # 2. <-- ВЫЗЫВАЕМ НАШУ ФУНКЦИЮ СКЛЕЙКИ (убираем огрызки меньше 300 символов)
    chunks = merge_small_chunks(raw_chunks, min_size=300)

    points = []

    for i, chunk_doc in enumerate(chunks):
        chunk_text = chunk_doc.page_content
        chunk_headers = chunk_doc.metadata

        # <-- НОВЫЕ СТРОЧКИ: Добавляем префикс passage: для документов
        vector = model.encode("passage: " + chunk_text).tolist()
        point_id = str(uuid.uuid4())

        # Формируем payload
        payload = {
            "text": chunk_text,
            "page": page_num,  # ГАРАНТИРОВАННО верная страница
            "keywords": page_keywords,
            "chunk_index": i
        }
        # Добавляем заголовки
        payload.update(chunk_headers)

        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"✅ Страница {page_num}: успешно загружено {len(points)} чанков.")


def run_vectorization():
    # 1. Получаем отсортированный список файлов
    files = [f for f in os.listdir(JSON_FOLDER) if f.startswith("page_") and f.endswith(".json")]
    files.sort(key=lambda x: int(x.split('_')[1]))

    for i, filename in enumerate(files):
        filepath = os.path.join(JSON_FOLDER, filename)
        page_num = int(filename.split('_')[1])

        print(f"\n📖 Обработка страницы {page_num}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            current_data = json.load(f)
            text_to_upload = current_data.get("refined_text", "")
            page_keywords = current_data.get("keywords", [])

        if i + 1 < len(files):
            next_page_num = page_num + 1
            next_filepath = os.path.join(JSON_FOLDER, files[i + 1])
            with open(next_filepath, 'r', encoding='utf-8') as f_next:
                next_data = json.load(f_next)
                next_text = next_data.get("refined_text", "")

                # Берем сырые 500 символов
                raw_overlap = next_text[:500]

                # <-- НОВЫЕ СТРОЧКИ: Учитываем списки (;) и переносы строк
                # Ищем точку, !, ? или точку с запятой (;), после которых идет пробел или конец строки
                matches = list(re.finditer(r'[.!?;](?=\s|$)', raw_overlap))

                if matches:
                    # Отрезаем ровно по этот знак включительно (+1)
                    last_punctuation = matches[-1].start()
                    overlap_text = raw_overlap[:last_punctuation + 1]
                else:
                    # Предохранитель 1: режем по последнему абзацу/переносу строки
                    last_newline = raw_overlap.rfind('\n')
                    if last_newline != -1:
                        overlap_text = raw_overlap[:last_newline]
                    else:
                        # Предохранитель 2 (Абсолютный): режем по последнему пробелу, чтобы не рвать слова
                        last_space = raw_overlap.rfind(' ')
                        if last_space != -1:
                            overlap_text = raw_overlap[:last_space]
                        else:
                            overlap_text = raw_overlap

                text_to_upload += f"\n\n--- НАЧАЛО СТРАНИЦЫ {next_page_num} ---\n\n" + overlap_text

        # 3. Отправляем в базу
        process_and_upload(text_to_upload, page_keywords, page_num)


if __name__ == "__main__":
    run_vectorization()
