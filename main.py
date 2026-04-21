import re
from core.parser import parse_pdf_to_markdown
from core.refiner import refine_medical_chunk
from utils.abbreviations import force_expand_abbreviations
from step3_saver import save_chunk_to_folder

BOOK_PATH = "./data/Артериальная гипертония 2024 РКО.pdf"


def run_pipeline(start_p, end_p):
    print(f"🚀 Запуск текстового конвейера (Страницы {start_p}-{end_p})")

    # 1. Получаем только текст
    raw_markdown = parse_pdf_to_markdown(BOOK_PATH, start_p, end_p)

    # 2. Делим на чанки
    chunks = re.split(r'\n(?=## )', raw_markdown)

    for i, chunk in enumerate(chunks):
        if not chunk.strip(): continue

        # 3. Расшифровка и чистка
        expanded = force_expand_abbreviations(chunk)
        refined_data = refine_medical_chunk(expanded)

        if refined_data:
            save_chunk_to_folder(refined_data, f"chunk_{start_p}_{i + 1:03d}.json")


if __name__ == "__main__":
    run_pipeline(188, 189)