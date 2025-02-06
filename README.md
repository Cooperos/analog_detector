# Analog Detector  
Репозиторий посвящен системе определения показаний на аналоговых приборах, разработанной в рамках чемпионата **REASkills 2025**. Проект использует современные технологии компьютерного зрения для точного анализа и классификации изображений аналоговых приборов.  

---

## Quick Start  
1. Клонируйте репозиторий:  
   ```bash
   git clone https://github.com/Cooperos/analog_detector
   ```
* Скачайте данные для обучения и поместите их в директорию **Images** [Ссылка](https://yandex.ru/video/preview/9791159952283275697)
* Запустите скрипт для обучения модели и получения предсказаний:
    ```bash
    python my_dag_airflow.py
    ```

# Технологии репозитория
В проекте используются следующие технологии:
* **YOLO Detect & YOLO Classificator:** Для обнаружения объектов и классификации изображений. [Документация YOLO](https://docs.ultralytics.com/ru/models/)
* **TensorFlow CNN:** Для построения и обучения сверточных нейронных сетей. [Документация TensorFlow](https://www.tensorflow.org/?hl=ru)

![YOLO arch](images/gre2kvy6az2jboykgg_jvazb3jg.jpeg)

# Принцип работы системы
Работа системы разделена на несколько этапов:
1. **Загрузка изображений:** Импорт изображений аналоговых приборов.
2. **Аугментация изображений:** Предобработка и улучшение изображений для повышения качества обучения.
3. **Обучение модели классификации:** Обучение модели на подготовленных изображениях.
4. **Предсказание:** Получение предсказаний на валидационной выборке.

# Планы по улучшению

1. **Улучшение качества данных:** Расширение датасета и улучшение аугментации изображений.
2. **Оптимизация моделей:** Подбор более эффективных архитектур и гиперпараметров.
3. **Интеграция с API:** Создание API для удобного взаимодействия с системой.

# Лицензия

Проект распространяется под лицензией [MIT](https://ru.wikipedia.org/wiki/%D0%9B%D0%B8%D1%86%D0%B5%D0%BD%D0%B7%D0%B8%D1%8F_MIT).
