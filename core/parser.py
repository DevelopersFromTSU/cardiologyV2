import os
import fitz
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


def parse_pdf_pro(pdf_path, start_page=1, end_page=1):
    temp_pdf = "temp_slice.pdf"
    output_dir = "temp_images"
    os.makedirs(output_dir, exist_ok=True)

    with fitz.open(pdf_path) as src:
        with fitz.open() as dest:
            dest.insert_pdf(src, from_page=start_page - 1, to_page=end_page - 1)
            dest.save(temp_pdf)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.generate_picture_images = True  # Включаем захват картинок

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    result = converter.convert(temp_pdf)
    md_text = result.document.export_to_markdown()

    image_paths = []
    seen_dimensions = set()

    for i, picture in enumerate(result.document.pictures):
        if picture.image is not None:
            pil_img = picture.image.pil_image
            w, h = pil_img.size

            # Фильтрация по ширине и высоте
            if (w, h) not in seen_dimensions:
                seen_dimensions.add((w, h))
                img_path = os.path.join(output_dir, f"img_{start_page}_{i}.png")
                pil_img.save(img_path)
                image_paths.append(img_path)

    if os.path.exists(temp_pdf):
        os.remove(temp_pdf)

    return md_text, image_paths