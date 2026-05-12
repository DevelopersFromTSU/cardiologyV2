import re
import os
import time  # <-- НОВЫЙ ИМПОРТ
import json

from core.parser import parse_pdf_pro
from core.vision import describe_image
from core.refiner import refine_medical_chunk
from utils.abbreviations import force_expand_abbreviations

BOOK_PATH = "./data/Артериальная гипертония 2024 РКО.pdf"


def save_chunk_to_folder(chunk_data, filename):
    """
    Создает папку result и сохраняет туда данные в формате JSON.
    """
    folder_name = "result"

    # Автоматически создаем папку, если её нет
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"📁 Создана папка: {folder_name}/")

    file_path = os.path.join(folder_name, filename)

    with open(file_path, "w", encoding="utf-8") as file:
        # indent=4 делает файл читаемым, ensure_ascii=False сохраняет кириллицу
        json.dump(chunk_data, file, ensure_ascii=False, indent=4)

    print(f"✅ Результат сохранен: {file_path}")


def inject_vision_data(md_text, image_paths):
    for img_path in image_paths:
        description = describe_image(img_path)
        replacement = f"\n\n{description}\n\n"
        md_text = re.sub(r"(<!--\s*image\s*-->|!\[.*?\]\(.*?\))", replacement, md_text, count=1)
        if os.path.exists(img_path):
            os.remove(img_path)
    return md_text


def run_pipeline(start_p, end_p):
    # <-- НОВАЯ СТРОЧКА: Добавили цикл для прохода по каждой странице отдельно
    for current_page in range(start_p, end_p + 1):
        print(f"\n🔄 Начинаем обработку страницы {current_page}...")

        # Передаем current_page вместо всего диапазона сразу
        raw_markdown, image_paths = parse_pdf_pro(BOOK_PATH, current_page, current_page)
        full_text = inject_vision_data(raw_markdown, image_paths)
        full_text_expanded = force_expand_abbreviations(full_text)

        refined_page = refine_medical_chunk(full_text_expanded)

        if refined_page:
            save_chunk_to_folder(refined_page, f"page_{current_page}_full.json")
            print(f"✅ Страница {current_page} успешно сохранена.")

        # <-- НОВЫЕ СТРОЧКИ: Проверяем, не последняя ли это страница, и делаем паузу
        if current_page < end_p:
            delay_minutes = 1
            delay_seconds = delay_minutes * 60
            print(f"⏳ Пауза {delay_minutes} мин. для сброса лимитов API Gemini...")
            time.sleep(delay_seconds)


if __name__ == "__main__":
    run_pipeline(30, 109)
