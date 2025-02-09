import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import cv2  # Импортируем OpenCV
from PIL import Image  # Импортируем PIL для работы с изображениями


class CNNModel(nn.Module):
    """Определение сверточной нейронной сети."""


    def __init__(self):
        super(CNNModel, self).__init__()
        # Первый свертка + активация + подвыборка
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 32 фильтра
        self.bn1 = nn.BatchNorm2d(32)  # Нормализация
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Подвыборка
        # Второй свертка + активация + подвыборка
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64 фильтра
        self.bn2 = nn.BatchNorm2d(64)  # Нормализация
        # Третий свертка
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 128 фильтров
        self.bn3 = nn.BatchNorm2d(128)  # Нормализация
        # Полносвязные слои
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # Измените размер в зависимости от входного размера
        self.fc2 = nn.Linear(256, 47)  # Выходной слой для 47 классов
        self.dropout = nn.Dropout(0.5)  # Дропаут для регуляризации

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Свертка 1
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Свертка 2
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Свертка 3
        x = x.view(-1, 128 * 3 * 3)  # Преобразование в вектор для полносвязного слоя
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Применение дропаута
        x = self.fc2(x)
        return x

class Model:
    def __init__(self):
        """Инициализация модели и загрузка весов."""
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.ckpt')
        self.model = CNNModel()  # Инициализация модели
        self.model.load_state_dict(torch.load(model_path))  # Загрузка весов модели
        self.model.eval()  # Установка режима оценки

        # Определение трансформации для входных изображений
        #self.transform = transforms.Compose([
        #    transforms.ToTensor(),
        #    transforms.Normalize((0.1307,), (0.3081,))
        #])

        # Определение трансформации для входных изображений
        self.transform = transforms.Compose([
            # Поворот на 90 градусов
            transforms.Lambda(lambda img: transforms.functional.rotate(img, -90)),
            # Горизонтальное отражение
            transforms.Lambda(lambda img: transforms.functional.hflip(img)),
            # Преобразование в тензор
            transforms.ToTensor(),
            # Нормализация
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        label_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '../data/EMNIST/raw/emnist-balanced-mapping.txt')
        with open(label_path, 'r') as f:
            label_mapping = f.readlines()

            # Создаем словарь соответствий
            self.label_dict = {}
            for entry in label_mapping:
                label, ascii_code = map(int, entry.split())
                self.label_dict[label] = chr(ascii_code)

    def preprocess_image(self, image_pixels):
        # Удаляем квадратные скобки и пробелы из строки
        image_pixels = image_pixels.strip()[1:-1]  # Удаляем '[' и ']'

        # Преобразование строки пикселей в массив NumPy
        image = np.array(list(map(int, image_pixels.split(',')))).reshape((28, 28))

        # Применение сглаживания (гауссов фильтр)
        smoothed = cv2.GaussianBlur(image.astype(np.uint8), (5, 5), 0)

        # Бинаризация изображения
        _, binary = cv2.threshold(smoothed, 128, 255, cv2.THRESH_BINARY_INV)

        # Определение структурного элемента
        kernel = np.ones((3, 3), np.uint8)

        # Эрозия
        eroded = cv2.erode(binary, kernel, iterations=1)

        # Дилатация
        dilated = cv2.dilate(eroded, kernel, iterations=1)

        # Преобразование обратно в тензор PyTorch и нормализация
        processed_image = torch.tensor(dilated).float().unsqueeze(0)  # Добавление размерности для батча
        processed_image = (processed_image - 127.5) / 127.5  # Нормализация

        return processed_image

    def predict(self, image):
        """
        Метод для предсказания класса на основе входного изображения.
        Обработка входного изображения с использованием методов Eroding,
        Dilating и Smoothing Images.

        :param image: Входное изображение в формате тензора или PIL Image.
        :return: Предсказанный символ.
        """
        image = self.preprocess_image(image)

        with torch.no_grad():  # Отключаем градиенты для повышения производительности
            output = self.model(image)
            _, predicted_class = torch.max(output.data, 1)

            predicted_label = predicted_class.item()
            pred_symbol = self.label_dict[predicted_label]
            print(pred_symbol)

            return pred_symbol