import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from google.colab import drive

# Configuración
img_shape = (64, 64, 3)  # Tamaño de las imágenes (64x64 píxeles, 3 canales de color)
latent_dim = 100         # Dimensión del vector de ruido

# Definir el generador
def build_generator(latent_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(np.prod(img_shape), activation='tanh'))
    model.add(layers.Reshape(img_shape))
    return model

# Definir el discriminador
def build_discriminator(img_shape):
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=img_shape))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Crear los modelos
generator = build_generator(latent_dim)
discriminator = build_discriminator(img_shape)

# Compilar el discriminador
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])

# Crear y compilar la GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

# Función para entrenar la GAN
def train_gan(epochs, batch_size, dataset):
    for epoch in range(epochs):
        # Entrenar el discriminador con imágenes reales
        idx = np.random.randint(0, dataset.shape[0], batch_size)
        real_images = dataset[idx]
        real_labels = np.ones((batch_size, 1))
        
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        
        # Entrenar el discriminador con imágenes falsas
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_images = generator.predict(noise)
        fake_labels = np.zeros((batch_size, 1))
        
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        
        # Promediar las pérdidas del discriminador
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Entrenar el generador para engañar al discriminador
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real_labels)
        
        # Mostrar el progreso
        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")
            if epoch % 1000 == 0:
                show_generated_image(generator)
                
# Función para mostrar una imagen generada
def show_generated_image(generator):
    noise = np.random.normal(0, 1, (1, latent_dim))
    gen_image = generator.predict(noise)[0]
    gen_image = 0.5 * gen_image + 0.5  # Reescalar de [-1, 1] a [0, 1]
    plt.imshow(gen_image)
    plt.axis('off')
    plt.show()

# Función para cargar las imágenes desde un directorio
def load_images_from_directory(directory, img_size=(64, 64)):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img)
            img_array = (img_array / 127.5) - 1  # Normalizar a rango [-1, 1]
            images.append(img_array)
    return np.array(images)

from google.colab import drive
drive.mount('/content/drive')
print("Drive montado a maquina Google Cloud")

import os
base_path = '/content/drive/MyDrive'
#print("Contenido de MyDrive:")
#print(os.listdir(base_path))

folder_path = '/content/drive/MyDrive/IA/archive/brain-tumor-mri-dataset'
print("Contenido de la carpeta:")
#print(os.listdir(folder_path))

# Directorio donde están las imágenes en tu Google Drive
data_dir = '/content/drive/MyDrive/IA/archive/brain-tumor-mri-dataset/glioma'


# Cargar las imágenes desde Google Drive
dataset = load_images_from_directory(data_dir, img_size=(64, 64))
print(f"Total imágenes cargadas: {dataset.shape[0]}")

# Entrenar la GAN Generative Adversarial Networks
print("Iniciando entrenamiento de Red Generativa")
train_gan(epochs=10000, batch_size=64, dataset=dataset)

