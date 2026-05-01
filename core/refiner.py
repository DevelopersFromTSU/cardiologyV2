import re
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

FOLDER_ID = os.getenv("FOLDER_ID")
API_KEY = os.getenv("API_KEY")


def normalize_markdown_table(text):
    text = re.sub(r'\|[-]{5,}\|', '|---|', text)
    text = re.sub(r'\s*\|\s*', ' | ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text


def minify_markdown_table_dashes(text):
    """Жестко пересобирает разделители таблиц на |-| с помощью Python, без нейросетей."""
    if not text:
        return text

    lines = text.split('\n')
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Ищем строку-разделитель (содержит только пробелы, |, -, :)
        if '|' in stripped and '-' in stripped and re.match(r'^[\s\|\-:]+$', stripped):
            # Считаем количество столбцов по символу |
            cols = stripped.count('|') - 1
            if cols > 0:
                # Собираем новую строку ровно с одним тире
                lines[i] = '|' + '|'.join(['-'] * cols) + '|'

    return '\n'.join(lines)


def refine_medical_chunk(chunk_text):
    chunk_text = normalize_markdown_table(chunk_text)
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {API_KEY}"
    }

    prompt = """
    Ты — строгий технический редактор медицинской документации. Твоя единственная задача: исправить опечатки и грамматику в тексте, СТРОГО сохранив исходную Markdown-разметку.
    1. ЗАГОЛОВКИ: ОРИГИНАЛЬНЫЕ заголовки копируй без изменений.
    2. ИНТЕГРАЦИЯ: Вставь данные из блоков [[MEDICAL_ALGORITHM...]] сразу под соответствующий заголовок, убрав сами маркеры.
    3. НИКАКОЙ ОТСЕБЯТИНЫ: Только сухой Markdown.

    ВЕРНИ СТРОГИЙ JSON:
    {
        "refined_text": "исправленный текст в Markdown",
        "category": "диагностика/лечение",
        "keywords": ["ключ1", "ключ2"]
    }
    """

    body = {
        "modelUri": f"gpt://{FOLDER_ID}/yandexgpt/latest",
        "completionOptions": {
            "stream": False,
            "temperature": 0.1,
            "maxTokens": "8000"
        },
        "messages": [
            {"role": "system", "text": prompt},
            {"role": "user", "text": chunk_text}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=body)
        if response.status_code != 200:
            print(f"❌ Детали ошибки от Яндекса: {response.text}")
        response.raise_for_status()

        raw_result = response.json()['result']['alternatives'][0]['message']['text']
        clean_json_str = raw_result.strip().replace('```json', '').replace('```', '')

        data = json.loads(clean_json_str)

        # Подключаем Лайт-модель для финальной зачистки тире
        if "refined_text" in data:
            data["refined_text"] = minify_markdown_table_dashes(data["refined_text"])

        return data

    except Exception as e:
        print(f"❌ Ошибка при обработке чанка в YandexGPT: {e}")
        return None