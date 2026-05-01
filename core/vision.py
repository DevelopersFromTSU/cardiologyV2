import os
from PIL import Image
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Убедись, что прокси настроены, если работаешь через них
os.environ['HTTPS_PROXY'] = os.getenv('HTTPS_PROXY', '')
os.environ['HTTP_PROXY'] = os.getenv('HTTP_PROXY', '')

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def describe_image(image_path):
    if not os.path.exists(image_path):
        return ""

    # Используем новейшую модель из твоего списка
    model_id = "gemini-2.5-pro"

    prompt = (
        "Ты — профессиональный медицинский оцифровщик. Твоя задача: перенести текст с картинки "
        "в Markdown с ПРЕДЕЛЬНОЙ точностью. ЗАПРЕЩЕНО использовать внешние знания, "
        "оцифровывай только то, что видишь на изображении.\n\n"
        "1. АЛГОРИТМЫ: Используй списки и стрелки `->`. Опиши КАЖДЫЙ блок (включая розовые и синие).\n"
        "2. ТЕРМИНЫ: Тщательно распознавай буквы. Не путай 'Инициацию' с 'Федерацией'.\n"
        "3. ЗАГОЛОВКИ: Все заголовки внутри ответа пиши через '###'.\n"
        "4. ТАБЛИЦЫ: Используй стандартный Markdown (|---|).\n"
        "5. БЕЗ ЛИШНЕГО: Не добавляй текст, которого нет на картинке."
    )

    try:
        with Image.open(image_path) as img:
            response = client.models.generate_content(
                model=model_id,
                contents=[prompt, img]
            )
            # В версии 2.5 доступ к тексту такой же
            return f"\n\n{response.text}\n\n"
    except Exception as e:
        print(f"⚠️ Ошибка Gemini API при оцифровке: {e}")
        return ""