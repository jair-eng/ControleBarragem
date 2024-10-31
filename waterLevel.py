import  cv2
import numpy as np

def detect_water_level(image, model, threshold=0.5):
    # Redimensionar e normalizar a imagem para entrada do modelo
    input_img = cv2.resize(image, (128, 128)) / 255.0
    input_img = input_img[np.newaxis, ...]  # Adiciona dimensão de batch

    # Prever a máscara
    predicted_mask = model.predict(input_img)[0, :, :, 0]

    # Aplicar um limiar para binarizar a máscara
    binary_mask = (predicted_mask > threshold).astype(np.uint8)

    # Calcula a proporção de área coberta pela água (1s na máscara)
    water_coverage = np.sum(binary_mask) / binary_mask.size
    return water_coverage