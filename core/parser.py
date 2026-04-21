import os
import fitz
import re
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


def parse_pdf_to_markdown(pdf_path, start_page=1, end_page=1):
    """
    Парсит выбранные страницы PDF в Markdown.
    """
    temp_pdf = "temp_slice.pdf"

    # 1. Сначала создаем временный файл только с нужными страницами
    # Это "золотой стандарт", так как Docling не будет сканировать лишнее
    with fitz.open(pdf_path) as src:
        with fitz.open() as dest:
            dest.insert_pdf(src, from_page=start_page - 1, to_page=end_page - 1)
            dest.save(temp_pdf)

    # 2. Настраиваем Docling на работу с этим срезом
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True  # Включаем OCR для сложных таблиц

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    print(f"⌛ Парсинг текста и таблиц (стр. {start_page}-{end_page})...")
    result = converter.convert(temp_pdf)
    md_text = result.document.export_to_markdown()

    # 3. Чистим текст от тегов картинок и удаляем временный файл
    md_text = re.sub(r"", "", md_text)

    if os.path.exists(temp_pdf):
        os.remove(temp_pdf)

    return md_text