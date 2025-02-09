import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np


# Определение модели
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Полносвязные слои
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 47)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01))
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Model:
    def __init__(self):
        """Инициализация модели и загрузка весов."""
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_08_2.ckpt')
        self.model = CNNModel()  # Инициализация модели
        self.model.load_state_dict(torch.load(model_path))  # Загрузка весов модели
        self.model.eval()  # Установка режима оценки

        # Определение трансформации для входных изображений
        self.transform = transforms.Compose([
            transforms.ToTensor(),
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


    def predict(self, image):

        # Проверяем тип входного изображения
        if isinstance(image, torch.Tensor):
            # Если это тензор и у него 2 измерения (28x28), добавляем размерность канала и батча
            if image.dim() == 2:
                image_tensor = image.unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]
            elif image.dim() == 3:
                image_tensor = image.unsqueeze(0)  # [1, 1, 28, 28]
            else:
                raise ValueError("Input tensor must have shape [height, width] or [channels, height, width]")
        else:
            # Применяем трансформацию к изображению
            image_tensor = self.transform(image)
            # Добавляем размерность батча (1)
            image_tensor = image_tensor.unsqueeze(0)  # [1, channels, height, width]

        print("Image tensor predict shape:", image_tensor.shape)  # Проверка формы тензора

        with torch.no_grad():  # Отключаем градиенты для повышения производительности
            output = self.model(image_tensor)  # Получаем выход модели
            _, predicted_class = torch.max(output.data, 1)  # Находим класс с максимальной вероятностью

            predicted_label = predicted_class.item()  # Получаем метку класса как целое число

            pred_symbol = self.label_dict[predicted_label]  # Получаем символ из словаря меток
            print(pred_symbol)

            return pred_symbol


    def predict_old(self, image):
        """
        Метод для предсказания класса на основе входного изображения.

        :param image: Входное изображение в формате тензора или PIL Image.
        :return: Предсказанный символ.
        """

        if isinstance(image, torch.Tensor):
            image_tensor = image.unsqueeze(
                0) if image.dim() == 2 else image  # Добавляем размерность батча если необходимо
        else:
            image_tensor = self.transform(image)  # Применяем трансформацию к изображению

            # Добавляем размерность батча (1)
            image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():  # Отключаем градиенты для повышения производительности
            output = self.model(image_tensor)  # Получаем выход модели
            _, predicted_class = torch.max(output.data, 1)  # Находим класс с максимальной вероятностью

            predicted_label = predicted_class.item()  # Получаем метку класса как целое число

            pred_symbol = self.label_dict[predicted_label]  # Получаем символ из словаря меток
            print(pred_symbol)

            return pred_symbol




# Пример использования:
if __name__ == "__main__":
    model_instance = Model()
    print("Model loaded and ready for predictions.")
