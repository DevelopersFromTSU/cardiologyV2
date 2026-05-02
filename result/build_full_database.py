import os
import json

# Указываем путь к папке с результатами обработки Gemini
JSON_FOLDER = "./"
OUTPUT_FILE = "full_book_refined.json"


def build_unified_json():
    all_pages_text = ""

    if not os.path.exists(JSON_FOLDER):
        print(f"⚠️ Папка '{JSON_FOLDER}' не найдена.")
        return

    # 1. Собираем и сортируем файлы по номеру страницы
    files = [f for f in os.listdir(JSON_FOLDER) if f.startswith("page_") and f.endswith(".json")]
    files.sort(key=lambda x: int(x.split('_')[1]))

    print(f"📦 Склеиваем {len(files)} страниц...")

    for filename in files:
        filepath = os.path.join(JSON_FOLDER, filename)
        page_num = filename.split('_')[1]

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            page_content = data.get("refined_text", "")
            # Добавляем разметку страницы для сплиттера
            all_pages_text += f"\n\n# Страница {page_num}\n\n{page_content}"

    # 2. Просто сохраняем результат в файл
    final_data = {
        "refined_text": all_pages_text,
        "category": "Медицинский справочник"
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

    print(f"✅ Файл '{OUTPUT_FILE}' успешно создан.")


if __name__ == "__main__":
    build_unified_json()