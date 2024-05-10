import tensorflow as tf
import matplotlib.pyplot as plt
import os
# Definir o caminho para a pasta de treinamento
train_dir = 'data-students/TRAIN'

# Criar um gerador de dados de imagem com data augmentation
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

# Carregar as imagens a partir da pasta de treinamento
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical'  # Mudar para 'categorical' para multiclasse
)

num_classes = train_generator.num_classes  # Obter o número de classes

# Criar um modelo de base usando a API funcional do TensorFlow
inputs = tf.keras.Input(shape=(32, 32, 3))
x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(128, 3, activation='relu')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)  # Mudar para 'softmax' para multiclasse

model = tf.keras.Model(inputs, outputs)

# Compilar o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Mudar para 'categorical_crossentropy' para multiclasse
              metrics=['accuracy'])

# Treinar o modelo
model.fit(train_generator, epochs=100)

# Definir o caminho para o diretório onde deseja salvar as imagens aumentadas
save_dir = 'augmented_images'

# Verificar se o diretório de salvamento existe, se não, criá-lo
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Iterar sobre o gerador de dados e salvar manualmente as imagens geradas
for i, batch in enumerate(train_generator):
    images, _ = batch  # Obter apenas as imagens, ignorando os rótulos
    for j, image in enumerate(images):
        filename = f'{save_dir}/augmented_image_{i * len(images) + j}.jpeg'
        tf.keras.preprocessing.image.save_img(filename, image)