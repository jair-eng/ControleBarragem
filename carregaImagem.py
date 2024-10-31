import os
import json
import numpy as np
from PIL import Image
import cv2  # Certifique-se de que OpenCV está instalado

def load_images_and_masks(image_dir, mask_dir, img_size=(128, 128)):
    # List all images and masks
    all_files = os.listdir(image_dir)
    print("Todos os arquivos no diretório de imagens:", all_files)

    image_filenames = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    mask_filenames = [f for f in os.listdir(mask_dir) if f.endswith('.json')]

    print("Imagens disponíveis:", image_filenames)
    print("Máscaras disponíveis:", mask_filenames)

    mask_dict = {mask_file[:-5]: mask_file for mask_file in mask_filenames}  # Remove '.json'

    x_train = []  # Inicializando a lista para imagens
    y_train = []  # Inicializando a lista para máscaras

    for img_file in image_filenames:
        base_name = img_file[:-4]  # Remove a extensão

        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path).resize(img_size)
        img = np.array(img) / 255.0  # Normalizando para [0, 1]

        if base_name in mask_dict:
            mask_path = os.path.join(mask_dir, mask_dict[base_name])
            try:
                with open(mask_path, 'r') as f:
                    mask_data = json.load(f)
                    print(f"Dados da máscara carregada de {mask_path}: {mask_data}")  # Debug

                # Criar uma máscara vazia
                mask = np.zeros(img_size, dtype=np.float32)

                # Processar cada forma
                for shape in mask_data.get('shapes', []):
                    points = shape['points']
                    points = np.array(points, dtype=np.int32)  # Convertendo para int para usar no OpenCV

                    # Desenhar o polígono na máscara
                    cv2.fillPoly(mask, [points], 1)  # Preencher o polígono com 1

                # Normalizando a máscara
                mask = mask / np.max(mask) if np.max(mask) > 0 else mask

                x_train.append(img)
                y_train.append(mask)

            except json.JSONDecodeError:
                print(f'Erro ao decodificar JSON de máscara {mask_path}.')
            except Exception as e:
                print(f'Erro ao carregar máscara {mask_path}: {e}')
        else:
            print(f'Máscara não encontrada para {base_name}')

    x_train = np.array(x_train)
    y_train = np.array(y_train)


    return x_train, y_train

# Para executar a função, você pode fazer isso no seu main()



