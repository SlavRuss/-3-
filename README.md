# Угадай кота - классификатор пород кошек

Веб-приложение для определения породы кошки по фотографии с использованием нейронной сети.

📌 О проекте

Приложение позволяет загрузить фотографию кошки и определяет её породу с помощью предварительно обученной модели машинного обучения. Результат отображается с указанием точности предсказания и описанием породы.

🛠 Технологии

- Backend: Python, Flask
- Frontend: HTML5, CSS3, JavaScript
- Машинное обучение: TensorFlow/Keras 
- Обработка изображений: Pillow

📂 Структура проекта

Ит.проэкт Свиста (3 курс)/

├── .venv/ # Виртуальное окружение Python

├── static/ # Статические файлы

│ ├── css/ # Стили

│ │ └── cat.css # Основные стили приложения

│ ├── images/ # Изображения пород кошек

│ │ ├── proto-cat.png # Изображение по умолчанию

│ │ ├── Абиссинская кошка.jpg

│ │ └── ... # Остальные изображения пород

│ ├── js/ # JavaScript файлы

│ └── uploads/ # Загруженные пользователями изображения

├── templates/ # Шаблоны HTML

│ ├── index.html # Главная страница

│ └── result_partial.html # Частичный шаблон результатов

├── app.py # Основное Flask-приложение

├── cat_classifier_model.keras # Обученная модель классификации

├── model_cat.ipynb # Ноутбук с обучением модели

└── requirements.txt # Зависимости Python


🛸 Вариант модели cat_classifier_model.keras с более 1000 изображений

🚀 Запуск проекта

1. Установка зависимостей:
pip install -r requirements.txt

2. Запуск сервера:
python app.py

3. Откройте в браузере:
http://localhost:5000

🔧 Настройка
Перед запуском убедитесь, что:
Установлен Python 3.8+
Все зависимости из requirements.txt установлены
В папке static/images/ присутствуют все изображения пород
Модель cat_classifier_model.keras находится в корне проекта

🌟 Особенности
Адаптивный дизайн
Плавные анимации и переходы
Возможность загружать фотографии без перезагрузки страницы
Подробное описание каждой породы кошек
Индикатор точности предсказания

📚 Поддерживаемые породы
Приложение определяет 10 пород кошек:
Бенгальская кошка
Ориентальная кошка
Русская голубая
Сноу-шу
Дворовая кошка
Абиссинская кошка
Британская кошка
Мейн-кун
Шотландская вислоухая
Сфинкс
