import requests
import json
import os
from dotenv import load_dotenv

# Загружаем переменные из .env
load_dotenv()

FOLDER_ID = os.getenv("FOLDER_ID")
API_KEY = os.getenv("API_KEY")


def refine_medical_chunk(chunk_text):
    """
    Финальная очистка медицинского текста после локального Layout-парсинга.
    Фокус: Markdown-таблицы, склонение терминов и удаление артефактов.
    """
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {API_KEY}"
    }

    # Обновленный лаконичный промпт без лишнего шума
    prompt = """
            Ты — ведущий медицинский редактор. Твоя задача — финализировать текст после оцифровки.

            ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:
            1. СОХРАНЕНИЕ ТАБЛИЦ: Все Markdown-таблицы (|---|) должны остаться нетронутыми. Не превращай их в обычный текст.
            2. ТЕРМИНОЛОГИЯ: Убедись, что все расшифрованные аббревиатуры (АГ, АД и др.) грамматически согласованы в предложениях.
            3. ОЧИСТКА: Удали технические артефакты оцифровки (повторяющиеся заголовки, номера страниц), если они разорвали предложение.

            ВЕРНИ СТРОГИЙ JSON:
            {
                "refined_text": "исправленный текст в Markdown",
                "category": "диагностика/лечение/прочее",
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
        response.raise_for_status()

        # Извлекаем текст из ответа Яндекса и парсим как JSON
        raw_result = response.json()['result']['alternatives'][0]['message']['text']

        # Очистка от возможных markdown-оберток нейросети (```json ... ```)
        clean_json_str = raw_result.strip().replace('```json', '').replace('```', '')
        return json.loads(clean_json_str)

    except Exception as e:
        print(f"❌ Ошибка при обработке чанка в YandexGPT: {e}")
        return None