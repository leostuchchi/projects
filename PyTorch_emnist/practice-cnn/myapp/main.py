import os
import torch
import matplotlib.pyplot as plt
from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
path = os.path.dirname(os.path.abspath(__file__))  # текущий путь к файлу
from myapp.model import Model
#from next_model import Model


import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


path = os.path.dirname(os.path.abspath(__file__))  # текущий путь к файлу

# load model
model = Model()

# app
app = FastAPI(title='Symbol detection', docs_url='/docs')

# api
@app.post('/api/predict')
def predict(image: str = Body(..., description='image pixels list')):
    image = torch.tensor(list(map(int, image[1:-1].split(',')))).reshape((28, 28))
    image = image.float()
    print("Image tensor api shape:", image.shape)  # Проверка формы тензора
    # Отображение изображения
    #plt.imshow(image.detach().numpy(), cmap='gray')
    #plt.title('Input Image')
    #plt.axis('off')  # Отключить оси
    #plt.show()
    pred = model.predict(image)
    return {'prediction': pred}



static_path = os.path.join(path, '../static')  # путь к папке 'static'
# статические файлы
app.mount('/', StaticFiles(directory=static_path, html=True), name='static')