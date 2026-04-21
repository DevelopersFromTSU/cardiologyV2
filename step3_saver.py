import os
import json

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