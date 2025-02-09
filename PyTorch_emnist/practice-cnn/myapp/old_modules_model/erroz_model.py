import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Определение модели
class ImprovedCNNModel(nn.Module):
    def __init__(self):
        super(ImprovedCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Убедитесь в правильном размере
        self.fc2 = nn.Linear(128, 47)  # Для 47 классов

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Model:
    def __init__(self):
        """Инициализация модели и загрузка весов."""
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_a_2.ckpt')#'model_rot_f.ckpt')
        self.model = ImprovedCNNModel()  # Инициализация модели
        self.model.load_state_dict(torch.load(model_path))  # Загрузка весов модели
        self.model.eval()  # Установка режима оценки

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
        # Преобразование строки пикселей в массив NumPy
        #image = np.array(list(map(int, image_pixels[1:-1].split(',')))).reshape((28, 28))
        image = image_pixels.numpy()

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

        # Преобразование обратно в тензор PyTorch и добавление размерностей
        processed_image = torch.tensor(dilated).float()  # Преобразование в тензор
        processed_image = processed_image.unsqueeze(0).unsqueeze(0)  # Добавление размерностей: [1, 1, 28, 28]

        return processed_image

    def predict(self, image):
        """
        Метод для предсказания класса на основе входного изображения.

        :param image: Входное изображение в формате строки пикселей.
        :return: Предсказанный символ.
        """
        print("Image tensor before shape:", image.shape)  # Проверка формы тензора

        image_tensor = self.preprocess_image(image)  # Применяем предобработку

        print("Image tensor after shape:", image_tensor.shape)  # Проверка формы тензора

        with torch.no_grad():  # Отключаем градиенты для повышения производительности
            output = self.model(image_tensor)
            _, predicted_class = torch.max(output.data, 1)


            predicted_label = predicted_class.item()
            pred_symbol = self.label_dict[predicted_label]
            print(pred_symbol)

            return pred_symbol
