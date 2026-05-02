import json
import os
from qdrant_db import process_and_upload_to_qdrant

# Путь к нашему склеенному файлу
UNIFIED_FILE = "./result/full_book_refined.json"

def upload_unified_data():
    if not os.path.exists(UNIFIED_FILE):
        print(f"⚠️ Файл '{UNIFIED_FILE}' не найден. Сначала запусти build_full_book.py.")
        return

    print(f"📖 Читаем объединенный файл {UNIFIED_FILE}...")
    with open(UNIFIED_FILE, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    print("🚀 Начинаем нарезку на умные чанки и загрузку в Qdrant...")
    # Отправляем весь текст. Сплиттер в qdrant_db сам найдет заголовки '## Страница'
    process_and_upload_to_qdrant(full_data, page_num=0)
    print("✅ Все данные успешно загружены в базу.")

if __name__ == "__main__":
    upload_unified_data()