import tensorflow as tf
import os
from keras.preprocessing.image import load_img, img_to_array

# Definir o caminho para a pasta de treinamento
train_dir = 'data-students/TRAIN'

# Subpastas que devem ser aplicadas data augmentation
subfolders_augment = ['13', '38', '39', '44']

# Data augmentation settings
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Diretório para salvar as imagens aumentadas
save_dir = 'augmented_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Limite de imagens a serem geradas por subpasta
images_per_subfolder_limit = 50

# Processar apenas as subpastas especificadas
for subfolder in subfolders_augment:
    subfolder_path = os.path.join(train_dir, subfolder)
    if os.path.isdir(subfolder_path):
        image_count = 0  # Contador para limitar o número de imagens geradas
        for image_file in os.listdir(subfolder_path):
            if image_count >= images_per_subfolder_limit:
                break  # Se atingir o limite, interrompe o loop para essa subpasta
            image_path = os.path.join(subfolder_path, image_file)
            image = load_img(image_path, target_size=(32, 32))
            image = img_to_array(image)
            image = image.reshape((1,) + image.shape)
            image_generator = datagen.flow(image, batch_size=1)
            img = next(image_generator)[0]  # Obter a primeira imagem do batch
            filename = os.path.join(save_dir, f'augmented_{subfolder}_{image_file}')
            tf.keras.preprocessing.image.save_img(filename, img)
            image_count += 1  # Incrementar contador de imagens salvas
