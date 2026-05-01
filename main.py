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
    raw_markdown, image_paths = parse_pdf_pro(BOOK_PATH, start_p, end_p)

    # Инъекция визуальных данных до разбивки на чанки
    full_text = inject_vision_data(raw_markdown, image_paths)

    chunks = re.split(r'\n(?=# )', full_text)

    for i, chunk in enumerate(chunks):
        if not chunk.strip(): continue

        expanded = force_expand_abbreviations(chunk)
        refined_data = refine_medical_chunk(expanded)

        if refined_data:
            save_chunk_to_folder(refined_data, f"chunk_{start_p}_{i + 1:03d}.json")


if __name__ == "__main__":
    run_pipeline(188, 188)