import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Загружаем модель (ту же самую, что работает у нас в поиске RAG)
print("⏳ Загрузка модели E5 для семантического анализа...")
model = SentenceTransformer('intfloat/multilingual-e5-base')

# 2. Наши "опорные точки" (якоря)
# ВАЖНО: Для модели E5 мы обязаны добавлять префикс "passage: " к эталонным фразам
REFERENCE_DATA = {
    "blood_pressure": [
        "passage: артериальное давление",
        "passage: давление сто сорок",
        "passage: высокое давление",
        "passage: тонометр",
        "passage: цифры давления",
        "passage: гипертензия"
    ],
    "habits": [
        "passage: курю сигареты",
        "passage: пачка в день",
        "passage: стаж курения",
        "passage: алкоголь",
        "passage: вредные привычки",
        "passage: выпиваю алкоголь"
    ],
    "complaints": [
        "passage: боль в груди",
        "passage: давит за грудиной",
        "passage: жжение в сердце",
        "passage: стенокардия",
        "passage: дискомфорт в грудной клетке",
        "passage: сердце болит"
    ],
    "risk_factors": [
        "passage: физическая нагрузка",
        "passage: ходьба",
        "passage: спорт",
        "passage: активность",
        "passage: лишний вес",
        "passage: соленая пища"
    ]
}

# 3. Считаем векторы эталонов один раз при запуске программы
print("⏳ Подготовка эталонных векторов...")
anchor_embeddings = {}
for category, phrases in REFERENCE_DATA.items():
    # model.encode сразу выдает нужные математические векторы
    anchor_embeddings[category] = model.encode(phrases)


def analyze_e5(user_text):
    # ВАЖНО: Для E5 вопрос пользователя должен иметь префикс "query: "
    user_query = f"query: {user_text}"
    user_vec = model.encode([user_query])  # Получаем вектор пациента

    results = {}

    # Сравниваем вектор пациента со всеми фразами из базы
    for category, ref_vecs in anchor_embeddings.items():
        # Считаем косинусное сходство (от -1 до 1)
        similarities = cosine_similarity(user_vec, ref_vecs)[0]
        # Берем максимальное сходство (самую близкую по смыслу фразу)
        results[category] = np.max(similarities)

    # Находим победителя
    best_category = max(results, key=results.get)
    score = results[best_category]

    print(f"\n📢 Ввод: '{user_text}'")
    print(f"🎯 Итог E5: {best_category} (Сходство: {score:.4f})")

    # У моделей семейства E5 сходство обычно высокое, поэтому порог ставим жестче (около 0.82)
    if score > 0.82:
        print(f"✅ Данные направлены в раздел: {best_category}")
    else:
        print("❓ Слишком размыто. Нужно уточнение от пациента.")

    return best_category, score


# --- ТЕСТОВЫЙ ЗАПУСК ---
if __name__ == "__main__":
    analyze_e5("Я курю уже лет десять точно.")
    analyze_e5("Вчера начало сильно давить за грудиной при ходьбе.")
    analyze_e5("Обычно моё давление где-то сто сорок на девяносто.")