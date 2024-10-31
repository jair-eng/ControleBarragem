from Entrada import build_unet
from carregaImagem import load_images_and_masks
from waterLevel import detect_water_level
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image
import numpy as np

# Carregar imagens e máscaras
if __name__ == "__main__":
    x_train, y_train = load_images_and_masks(
        'C:/Users/Jair/PycharmProjects/ControleBarragem/venv/imgRiosf',
        'C:/Users/Jair/PycharmProjects/ControleBarragem/venv/mskRiosf'
    )
    print(f"Imagens carregadas: {x_train.shape}, Máscaras carregadas: {y_train.shape}")

# Parâmetros do modelo
input_shape = (128, 128, 3)  # Ajuste conforme necessário
unet_model = build_unet(input_shape)

# Compilação do modelo com ajuste da taxa de aprendizado
unet_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
unet_model.summary()

# Data augmentation para evitar overfitting
data_gen_args = dict(rotation_range=15,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     shear_range=0.1,
                     zoom_range=0.1,
                     horizontal_flip=True,
                     fill_mode='nearest')

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Adicionar a dimensão de canal às máscaras (y_train)
y_train = np.expand_dims(y_train, axis=-1)
# Função para gerar dados de treinamento
# Função para gerar dados de treinamento
def train_generator(image_gen, mask_gen):
    while True:
        x = next(image_gen)
        y = next(mask_gen)
        yield x, y

# Gerador de dados com augmentação para treinamento
seed = 42
image_generator = image_datagen.flow(x_train, batch_size=8, seed=seed)
mask_generator = mask_datagen.flow(y_train, batch_size=8, seed=seed)

# Gerador de dados para validação (sem augmentação)
val_image_datagen = ImageDataGenerator()
val_mask_datagen = ImageDataGenerator()
val_image_generator = val_image_datagen.flow(x_train, batch_size=8, seed=seed)
val_mask_generator = val_mask_datagen.flow(y_train, batch_size=8, seed=seed)

# Compilando geradores usando as funções definidas
train_gen = train_generator(image_generator, mask_generator)
val_gen = train_generator(val_image_generator, val_mask_generator)

# Configuração do early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


# Treinamento do modelo
unet_model.fit(
    train_gen,
    steps_per_epoch=len(x_train) // 8,
    epochs=10,
    validation_data=val_gen,
    validation_steps=len(x_train) // 8,
    callbacks=[early_stopping]
)

# Carregar e processar imagem para detecção de nível de água
image_path = 'C:/Users/Jair/PycharmProjects/ControleBarragem/venv/tests/11.jpg'
image = np.array(Image.open(image_path))

# Detectar nível de água usando o modelo já treinado
detect_water_level(image, unet_model, threshold=0.5)

