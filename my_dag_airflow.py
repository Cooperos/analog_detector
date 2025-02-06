from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
# import logging


from ultralytics import YOLO
import tensorflow  as tf
from tensorflow.python.keras import layers, models
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import numpy as np
import time
import datetime

import os

def list_directories(path):
    # Получаем список всех элементов в указанной директории
    items = os.listdir(path)
    # Фильтруем только директории
    directories = [item for item in items if os.path.isdir(os.path.join(path, item))]

    return directories

def list_files_in_directory(path):
    # Получаем список всех элементов в указанной директории
    items = os.listdir(path)
    # Фильтруем только файлы
    files = [os.path.join(path, item) for item in items if os.path.isfile(os.path.join(path, item))]

    return files

classes = list_directories("../images/Gauge_big/train/")
train_files = list_files_in_directory("../prepocessed_images/Gauge_big/train/")
test_files = list_files_in_directory("../prepocessed_images/Gauge_big/test/")
val_files = list_files_in_directory("../prepocessed_images/Gauge_big/val/")

class ModelTrainingPipeline:
    def __init__(self, model_type: str, **hyperparameters):
        """
        model_type - тип модели на выбор
        **hyperparameters - гиперпараметры для конкретной модели
        """
        self.model_types: dict = { # Мои выбранные 3 типа моделей 
            "CNN": hyperparameters.get("CNN", {}),
            "YOLO_CLASSIFY": hyperparameters.get("YOLO_CLASSIFY", {}),
            "YOLO_DETECT": hyperparameters.get("YOLO_DETECT", {}),
        }

        if model_type not in self.model_types:
            raise ValueError("Выбран неверный тип модели")

        self.model = None
        self.model_type = model_type
        self.hyperparameters = self.model_types[model_type]

        print(self.hyperparameters)

        return self.evaluate_model_by_type()

    def evaluate_model_by_type(self) -> list: # выполняет создание и обучение выбранной модели
        """
        Возвращает параметры в следующем порядке:
        1 - модель
        2 - словарь с метриками
        3 - время в секундах на эпоху
        4 - суммарное время выполнения в минутах
        """
        self.pipeline_start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.model_type == "CNN":
            self.create_cnn_model()
            train_dataset = self.create_dataset(train_files, classes, 8)
            test_dataset = self.create_dataset(test_files, classes, 8) # Преобразуем файлы в датасеты для CNN
            val_dataset = self.create_dataset(val_files, classes, 8)
            self.train_cnn_model(train_dataset, test_dataset, val_dataset)
        
        elif self.model_type == "YOLO_CLASSIFY":
            self.create_yolo_classify_model()
            self.evaluate_yolo_training()

        elif self.model_type == "YOLO_DETECT":
            self.create_yolo_decect_model()
            self.evaluate_yolo_training()
            
        self.time_per_epoch = (self.total_time * 60) / self.hyperparameters["epochs"]
        results = [self.model, {"accuracy": self.accuracy, "roc_auc": self.roc_auc, "f1": self.f1}, self.time_per_epoch, self.total_time]
        self.save_results_to_file(results)
        return results

    def save_results_to_file(self, results):
        filename = f"training_results_{self.model_type}_{self.pipeline_start_time}.txt"

        with open(filename, 'w') as f:
            f.write("Результаты обучения модели:\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
            f.write(f"Общее время обучения (мин): {self.total_time}\n")
        print(f"Результаты сохранены в файл: {filename}")

    def create_yolo_decect_model(self):
        # Создает модель YOLO для детекции
        self.model = YOLO("../models/yolov8s.pt", verbose=True)
        if not self.model:
            return False
        return True
    
    def create_yolo_classify_model(self):
        # Создает модель YOLO для детекции
        self.model = YOLO("../models/yolov8s-cls.pt", verbose=True)
        if not self.model:
            return False
        return True

    def evaluate_yolo_training(self): # YOLO проводит валидацию после каждой эпохи обучения или заданного количества итераций, чтобы оценить производительность модели на данных валидации
        # Запускает обучение любой из моделей YOLO
        start_time = time.time()
        self.results = self.model.train(
            data="data.yaml",
            imgsz=244,
            epochs=self.hyperparameters['epochs'],
            batch=self.hyperparameters['batch'],
            lr0=self.hyperparameters['lr0'],
            momentum=self.hyperparameters['momentum'],
            weight_decay=self.hyperparameters['weight_decay'],
            name="yolo8vn_detection",
        )
        self.total_time = (time.time() - start_time) / 60

        self.yolo_final_validation()

    def yolo_final_validation(self):
        val_results = self.model.val(data="data.yaml")
        print("Итоговые результаты валидации:")
        print(val_results)


        val_images, val_labels = val_files, classes
        predictions = self.model.predict(val_images)

        # Обработка предсказаний
        predicted_classes = np.array([np.argmax(pred) for pred in predictions])

        # Расчет метрик
        self.accuracy = accuracy_score(val_labels, predicted_classes)
        self.roc_auc = roc_auc_score(val_labels, predictions, multi_class='ovr')
        self.f1 = f1_score(val_labels, predicted_classes, average='macro')

        print(f"Итоговая точность: {self.accuracy}")
        print(f"ROC AUC Score: {self.roc_auc}")
        print(f"F1 Score (macro): {self.f1}")


    def create_cnn_model(self):
        # Создание простой CNN модели

        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.hyperparameters['img_size'], self.hyperparameters['img_size'], 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dense(self.hyperparameters['num_classes'], activation='softmax'))  # Для многоклассовой классификации
        # Компиляция модели
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyperparameters['lr']),
                           loss='sparse_categorical_crossentropy',  # Или 'categorical_crossentropy' в зависимости от формата меток
                           metrics=['accuracy'])
        
    def load_images_tf(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = img / 255.0 # нормализация
        return img
    
    def create_dataset(self, image_paths, labels, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(lambda x, y: (self.load_images_tf(x), y))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def train_cnn_model(self, train_data, test_data, val_data):
        # Метод для обучения CNN модели
        start_time = time.time()
        history = self.model.fit(train_data, validation_data=test_data, epochs=self.hyperparameters['epochs'], batch_size=self.hyperparameters['batch'])
        
        # Промежуточное тестирование CNN модели
        for epoch in range(self.hyperparameters['epochs']):
            print(f"Epoch {epoch + 1}/{self.hyperparameters['epochs']}")
            self.model.fit(train_data, validation_data=test_data, epochs=1, batch_size=self.hyperparameters['batch'])
            test_loss, test_acc = self.model.evaluate(test_data)
            print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

        self.total_time = (time.time() - start_time) / 60

        test_loss, test_acc = self.model.evaluate(val_data)
        print(f"Итоговая потеря на валидационной выборке: {test_loss}, Итоговая точность: {test_acc}")

        #Получение предсказаний для расчета метрик
        val_images, val_labels = next(iter(val_data))
        predictions = self.model.predict(val_images)
        predicted_classes = np.argmax(predictions, axis=1)

        # Расчет метрик
        self.accuracy = accuracy_score(val_labels, predicted_classes)
        self.roc_auc = roc_auc_score(val_labels, predictions, multi_class='ovr')
        self.f1 = f1_score(val_labels, predicted_classes, average='macro')

        print(f"Итоговая точность: {self.accuracy}")
        print(f"ROC AUC Score: {self.roc_auc}")
        print(f"F1 Score (macro): {self.f1}")




default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=30),
    'execution_timeout': timedelta(hours=13, minutes=30),
}

dag = DAG(
    'nightly_model_training',
    default_args=default_args,
    description='Ночное обучение моделей компьютерного зрения',
    schedule_interval='0 18 * * *',  # 18:00 ежедневно
    catchup=False,
    max_active_runs=1,
)

def get_hyperparameters(model_type):
    configs = {
        "CNN": {
            "img_size": 224,
            "num_classes": 1000,
            "epochs": 20,
            "batch": 64,
            "lr": 0.0001
        },
        "YOLO_CLASSIFY": {
            "epochs": 50,
            "batch": 32,
            "lr0": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005
        },
        "YOLO_DETECT": {
            "epochs": 100,
            "batch": 16,
            "lr0": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0001
        }
    }
    return configs.get(model_type, {})

def load_training_data():
    # Реализуйте загрузку и предобработку данных
    return {
        'train_files': [],
        'test_files': [],
        'val_files': [],
        'classes': []
    }

def train_model_wrapper(**kwargs):
    try:
        model_type = "CNN"  # Можно параметризировать
        
        # Загрузка данных
        data = load_training_data()
        
        # Получение гиперпараметров
        hyperparams = get_hyperparameters(model_type)
        
        # Инициализация и запуск пайплайна
        pipeline = ModelTrainingPipeline(
            model_type=model_type,
            train_files=data['train_files'],
            test_files=data['test_files'],
            val_files=data['val_files'],
            classes=data['classes'],
            **hyperparams
        )
        
        # logging.info("Обучение успешно завершено")
        return True
    
    except Exception as e:
        # logging.error(f"Ошибка обучения: {str(e)}")
        # Дополнительная обработка ошибок
        error_report = f"Error report:\n{str(e)}"
        with open(f"/reports/error_{datetime.now().isoformat()}.txt", "w") as f:
            f.write(error_report)
        return True  # Возвращаем успех чтобы не блокировать последующие выполнения

training_task = PythonOperator(
    task_id='model_training_task',
    python_callable=train_model_wrapper,
    provide_context=True,
    dag=dag,
)

if __name__ == "__main__":
    training_task
