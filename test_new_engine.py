import os

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from docling.document_converter import DocumentConverter

# Подтягиваем путь из твоих настроек
# В новом проекте мы ориентируемся на папку data/ или books/
BOOK_PATH = "./data/Артериальная гипертония 2024 РКО.pdf"


def test_new_engine():
    """
    Проверка локального Layout-парсинга (Docling).
    Скрипт проверяет наличие файла, создает папку для отладки
    и выводит результат в Markdown.
    """

    # 1. Проверка путей
    if not os.path.exists(BOOK_PATH):
        # Если папки books нет, проверим альтернативный путь в папке data/
        alt_path = os.path.join("data", os.path.basename(BOOK_PATH))
        if os.path.exists(alt_path):
            current_path = alt_path
        else:
            print(f"❌ Файл не найден. Убедись, что PDF лежит здесь: {BOOK_PATH}")
            return
    else:
        current_path = BOOK_PATH

    print(f"🚀 Запуск Docling для файла: {os.path.basename(current_path)}")
    print("⏳ При первом запуске загружаются нейросетевые модели (1-2 ГБ)...")

    try:
        # 2. Инициализация конвертера
        converter = DocumentConverter()

        # 3. Конвертация
        result = converter.convert(current_path)

        # 4. Экспорт в Markdown (целевой формат для медицины)
        markdown_text = result.document.export_to_markdown()

        # 5. Сохранение результата в папку debug
        debug_dir = "debug"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

        output_file = os.path.join(debug_dir, "layout_test.md")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_text)

        print("\n" + "=" * 40)
        print(f"✅ ТЕСТ ЗАВЕРШЕН УСПЕШНО!")
        print(f"📝 Результат сохранен в: {output_file}")
        print(f"📊 Объем извлеченного текста: {len(markdown_text)} символов.")
        print("=" * 40)

        print("\n💡 Превью (первые 300 символов):")
        print("-" * 20)
        print(markdown_text[:300] + "...")
        print("-" * 20)
        print("\nПроверь файл layout_test.md — там должны быть склеенные таблицы.")

    except Exception as e:
        print(f"❌ Ошибка при работе Docling: {e}")


if __name__ == "__main__":
    # Убедись, что библиотека установлена: pip install docling
    test_new_engine()