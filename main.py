import re
import os
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

        # Заменяем стандартные теги Docling на полученный текст
        md_text = re.sub(r"(<!--\s*image\s*-->|!\[.*?\]\(.*?\))", replacement, md_text, count=1)

        if os.path.exists(img_path):
            os.remove(img_path)
    return md_text


def run_pipeline(start_p, end_p):
    # 1. Получаем Markdown и картинки[cite: 3]
    raw_markdown, image_paths = parse_pdf_pro(BOOK_PATH, start_p, end_p)

    # 2. Вставляем данные из Vision
    full_text = inject_vision_data(raw_markdown, image_paths)

    # 3. Расширяем аббревиатуры по всей странице[cite: 8]
    full_text_expanded = force_expand_abbreviations(full_text)

    # 4. Отправляем ВСЮ страницу в рефайнер (без сплита)
    # ВАЖНО: Убедись, что модель 2.0 Flash потянет объем страницы (обычно да)
    refined_page = refine_medical_chunk(full_text_expanded)

    if refined_page:
        # Сохраняем как один большой файл страницы
        save_chunk_to_folder(refined_page, f"page_{start_p}_full.json")


if __name__ == "__main__":
    run_pipeline(188, 188)