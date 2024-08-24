import os
import cv2
import numpy as np
from pathlib import Path
from keras.models import load_model

# Caminho para o modelo treinado
model_path = Path('./data/modelo/modelo.h5')

# Caminho para a pasta com as imagens de teste
test_folder = Path('./data/teste/')

# Carregando o modelo treinado
model = load_model(str(model_path))

# Definindo as opções de redimensionamento e normalização
resize_dim = (224,224)
normalize = True

# Lista com o nome dos arquivos a serem classificados
file_names = [f for f in os.listdir(test_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Loop pelos arquivos de teste
for file_name in file_names:
    try:
        # Carregando a imagem de teste
        img = cv2.imread(str(test_folder / file_name))

        # Pré-processando a imagem
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        if normalize:
            normalized = equalized / 255.0
        else:
            normalized = equalized
        resized = cv2.resize(normalized, resize_dim)
        input_img = np.expand_dims(resized, axis=0)
        input_img = input_img.reshape(input_img.shape[0], input_img.shape[1], input_img.shape[2], 1)

        # Realizando a predição com o modelo treinado
        pred = model.predict(input_img)

        # Imprimindo o resultado da predição
        if pred > 0.5:
            print(f"{file_name}: drone")
        else:
            print(f"{file_name}: não é drone")
    except Exception as e:
        print(f"Erro ao processar a imagem {file_name}: {e}")
