import os
import json
from qdrant_db import process_and_upload_to_qdrant

JSON_FOLDER = "./result"


def upload_with_manual_overlap():
    # 1. Получаем отсортированный список файлов[cite: 12]
    files = [f for f in os.listdir(JSON_FOLDER) if f.startswith("page_") and f.endswith(".json")]
    files.sort(key=lambda x: int(x.split('_')[1]))

    for i, filename in enumerate(files):
        filepath = os.path.join(JSON_FOLDER, filename)
        page_num = int(filename.split('_')[1])

        print(f"📖 Обработка страницы {page_num}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            current_data = json.load(f)
            text_to_upload = current_data.get("refined_text", "")

        # 2. РЕАЛИЗАЦИЯ НАХЛЕСТА: заглядываем в следующий файл
        if i + 1 < len(files):
            next_page_num = page_num + 1
            next_filepath = os.path.join(JSON_FOLDER, files[i + 1])
            with open(next_filepath, 'r', encoding='utf-8') as f_next:
                next_data = json.load(f_next)
                next_text = next_data.get("refined_text", "")
                # Берем начало следующей страницы (например, первые 600 символов)
                overlap_text = next_text[:800]
                text_to_upload += f"\n\n--- НАЧАЛО СТРАНИЦЫ {next_page_num} ---\n\n" + overlap_text

        # 3. Отправляем в базу с ГАРАНТИРОВАННЫМ номером страницы[cite: 11]
        refined_payload = {
            "refined_text": text_to_upload,
            "category": current_data.get("category", "Медицинский справочник")
        }

        process_and_upload_to_qdrant(refined_payload, page_num=page_num)
        print(f"✅ Страница {page_num} загружена с нахлестом от следующей.")


if __name__ == "__main__":
    upload_with_manual_overlap()