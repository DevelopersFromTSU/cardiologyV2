import re
import os
import time  # <-- НОВЫЙ ИМПОРТ

from core.parser import parse_pdf_pro
from core.vision import describe_image
from core.refiner import refine_medical_chunk
from utils.abbreviations import force_expand_abbreviations
from step3_saver import save_chunk_to_folder

BOOK_PATH = "./data/Артериальная гипертония 2024 РКО.pdf"


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
            delay_minutes = 2
            delay_seconds = delay_minutes * 60
            print(f"⏳ Пауза {delay_minutes} мин. для сброса лимитов API Gemini...")
            time.sleep(delay_seconds)


if __name__ == "__main__":

    run_pipeline(184, 188)