import re
import requests
import json
import os
from dotenv import load_dotenv
from google import genai
from utils.abbreviations import force_expand_abbreviations
import time

load_dotenv()

os.environ['HTTPS_PROXY'] = os.getenv('HTTPS_PROXY', '')
os.environ['HTTP_PROXY'] = os.getenv('HTTP_PROXY', '')

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def clean_excessive_whitespace(text):
    if not text:
        return text
    # 1. Заменяем 3 и более переносов строк на стандартные 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 2. Убираем пробелы и табуляцию в конце строк
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    # 3. Заменяем 2+ пробела между словами на один, сохраняя отступы в начале строк.
    # Используем positive lookbehind (?<=\S), чтобы искать пробелы только ПОСЛЕ символов
    text = re.sub(r'(?<=\S)[ \t]{2,}', ' ', text)
    return text

# def minify_markdown_table_dashes(text):
#     if not text:
#         return text
#     lines = text.split('\n')
#     for i, line in enumerate(lines):
#         stripped = line.strip()
#         if '|' in stripped and '-' in stripped and re.match(r'^[\s\|\-:]+$', stripped):
#             cols = stripped.count('|') - 1
#             if cols > 0:
#                 lines[i] = '|' + '|'.join(['-'] * cols) + '|'
#     return '\n'.join(lines)


def refine_medical_chunk(chunk_text, max_retries=3):
    model_id = "gemini-2.5-flash"  # Рекомендую 2.0, так как на него у тебя настроены квоты

    sys_instr = (
        "Ты — строгий технический редактор и медицинский аналитик. "
        "Твоя задача:\n"
        "1. ИСПРАВЛЕНИЕ: Устрани опечатки и грамматические ошибки.\n"
        "2. ПРЕОБРАЗОВАНИЕ ТАБЛИЦ: Если в тексте есть Markdown-таблицы (|---|), ПЕРЕПИШИ их в виде логических цепочек со стрелками `->`. "
        "Каждая строка должна быть самодостаточной: сочетай заголовок строки, заголовок столбца и значение в одно предложение.\n"
        "3. СОХРАНЕНИЕ СТРУКТУРЫ: В уже существующих алгоритмах строго сохраняй вложенность списков и стрелки.\n"
        "4. ФОРМАТ: Верни результат СТРОГО в формате JSON."
    )

    json_prompt = """
    {
        "refined_text": "исправленный текст",
        "category": "диагностика/лечение",
        "keywords": ["ключ1"]
    }
    """

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                # ЗАМЕНИ expanded_text на chunk_text
                contents=f"Обработай следующий текст и верни его в формате {json_prompt}:\n\n{chunk_text}",
                # ИСПРАВЛЕНИЕ 2: Исправили отступы
                config={
                    "system_instruction": sys_instr,
                    "response_mime_type": "application/json",
                    "temperature": 0.1
                }
            )

            data = json.loads(response.text)

            if "refined_text" in data:
                # 2. Затем вычищаем весь мусорный пробел, сохраняя отступы списков
                data["refined_text"] = clean_excessive_whitespace(data['refined_text'])

            return data

        except Exception as e:
            # ИСПРАВЛЕНИЕ 3: Починили логику повторов
            print(f"⚠️ Ошибка Gemini (попытка {attempt + 1}/{max_retries}): {e}")
            time.sleep(15)  # Ждем перед следующей попыткой

    # Если цикл закончился, а return data не сработал
    print("❌ Не удалось обработать текст после всех попыток.")
    return None