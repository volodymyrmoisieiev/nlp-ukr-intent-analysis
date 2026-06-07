import re
from pathlib import Path

import joblib
import pandas as pd
import gradio as gr


# =========================
# 1. Налаштування
# =========================

MODEL_PATH = Path("models/intent_logreg_tfidf.joblib")


# =========================
# 2. Завантаження моделі
# =========================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Файл моделі не знайдено: {MODEL_PATH}. "
        "Спочатку збережи модель у notebook."
    )

artifacts = joblib.load(MODEL_PATH)

model = artifacts["model"]
tfidf = artifacts["tfidf"]
class_names = artifacts["class_names"]


# =========================
# 3. Очищення тексту
# =========================

def clean_text(text):
    text = str(text).lower()

    # Видалення посилань
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Видалення mentions
    text = re.sub(r"@\w+", " ", text)

    # Видалення цифр
    text = re.sub(r"\d+", " ", text)

    # Залишаємо українські, англійські літери та пробіли
    text = re.sub(r"[^a-zA-Zа-щА-ЩЬьЮюЯяЇїІіЄєҐґ\s]", " ", text)

    # Нормалізація пробілів
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================
# 4. Українські назви класів
# =========================

class_translation = {
    "Complaint / Dissatisfaction": "Скарга / незадоволення",
    "Gratitude / Positive Feedback": "Подяка / позитивний відгук",
    "Neutral Comment": "Нейтральний коментар",
    "Question / Request for Help": "Питання / прохання про допомогу",
    "Suggestion / Idea": "Пропозиція / ідея",
}


# =========================
# 5. Функція прогнозування
# =========================

def predict_intent(user_text):
    if not user_text or user_text.strip() == "":
        empty_df = pd.DataFrame(columns=["Клас", "Ймовірність"])
        return "Введіть текст повідомлення.", 0.0, empty_df

    cleaned_text = clean_text(user_text)

    if cleaned_text == "":
        empty_df = pd.DataFrame(columns=["Клас", "Ймовірність"])
        return "Після очищення текст порожній. Введіть інше повідомлення.", 0.0, empty_df

    text_vector = tfidf.transform([cleaned_text])

    predicted_class = model.predict(text_vector)[0]

    probabilities = model.predict_proba(text_vector)[0]
    confidence = float(probabilities.max())

    predicted_class_ua = class_translation.get(predicted_class, predicted_class)

    probabilities_df = pd.DataFrame({
        "Клас": [class_translation.get(cls, cls) for cls in model.classes_],
        "Ймовірність": probabilities
    })

    probabilities_df = probabilities_df.sort_values(
        by="Ймовірність",
        ascending=False
    )

    probabilities_df["Ймовірність"] = probabilities_df["Ймовірність"].round(3)

    result_text = f"Передбачений інтент: {predicted_class_ua}"

    return result_text, round(confidence, 3), probabilities_df


# =========================
# 6. Gradio інтерфейс
# =========================

demo = gr.Interface(
    fn=predict_intent,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Введіть користувацьке звернення українською мовою...",
        label="Текст звернення"
    ),
    outputs=[
        gr.Textbox(label="Результат класифікації"),
        gr.Number(label="Confidence"),
        gr.Dataframe(label="Ймовірності по класах")
    ],
    title="Intent Analysis україномовних користувацьких звернень",
    description=(
        "Демонстраційна система автоматично визначає інтент користувацького "
        "повідомлення українською мовою. Модель побудована на основі "
        "TF-IDF-векторизації та Logistic Regression."
    ),
    examples=[
        ["Не можу увійти в акаунт, допоможіть будь ласка"],
        ["Дякую, все працює добре, дуже зручний сервіс"],
        ["Було б добре додати темну тему в застосунок"],
        ["Чому моя заявка так довго обробляється?"],
        ["Просто залишаю коментар щодо роботи сервісу"]
    ],
)


# =========================
# 7. Запуск застосунку
# =========================

if __name__ == "__main__":
    demo.launch()