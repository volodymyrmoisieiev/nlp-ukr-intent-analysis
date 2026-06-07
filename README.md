# NLP Project

Навчальний NLP-проєкт для класифікації інтентів в україномовних користувацьких зверненнях.

## Що є в проєкті

- `EDA + моделі.ipynb` — EDA, підготовка ознак, навчання та оцінювання моделей.
- `models/intent_logreg_tfidf.joblib` — збережена модель `TF-IDF + Logistic Regression`.
- `models/model_metadata.json` — метадані моделі та базові метрики.

## Структура

```text
project/
├── EDA + моделі.ipynb
├── models/
│   ├── intent_logreg_tfidf.joblib
│   └── model_metadata.json
├── .gitignore
└── README.md
```

## Запуск

1. Створіть та активуйте віртуальне середовище Python.
2. Встановіть основні залежності:

```bash
pip install pandas numpy matplotlib scikit-learn joblib jupyter
```

3. Переконайтеся, що файл датасету `UAReviews.csv` доступний поруч із ноутбуком або оновіть шлях у ноутбуці.
4. Запустіть Jupyter:

```bash
jupyter notebook
```

5. Відкрийте `EDA + моделі.ipynb` і виконайте клітинки послідовно.

## Примітка

Датасет `UAReviews.csv` у цій папці зараз відсутній, але ноутбук очікує його на етапі завантаження даних.
