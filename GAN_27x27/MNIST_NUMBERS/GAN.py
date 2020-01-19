# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 12:22:20 2019

@author: Diego

Basado en: https://github.com/eriklindernoren/Keras-GAN/tree/master/gan
"""

#from __future__ import print_function, division
import os
import numpy as np
import matplotlib.pyplot as plt

# Descomentar si se encuentran problemas con tensor y --- depende de la versión tensor flow instalada
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizers import Adam

# Semilla para función random
#np.random.seed(10)

def load_minst_data():
    # load the data
    #x_train = np.load('Data/Data_Train.npy')
    #y_train = np.load('Data/Data_Labels.npy')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #y_train = np.squeeze(y_train); # activar solo para cifar 10 #*****IMPORTANTE*****#
    # normalize our inputs to be in the range[-1, 1] 
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    # convert x_train with a shape of (60000, 28, 28) to (60000, 784) so we have
    # 784 columns per row
    return (x_train, y_train)

def build_generator(img_shape, latent_dim):

    model = Sequential()

    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    model.summary()

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)

def build_discriminator(img_shape, optimizer):

    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    
    # Establece formato de entrada
    img = Input(shape=img_shape)
    validity = model(img)
    
    # Compilar discriminator
    Model2 = Model(img, validity)
    Model2.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

    return Model2

def build_gan(generator, discriminator, latent_dim, optimizer):
    # The generator takes noise as input and generates imgs
    z = Input(shape=(latent_dim,))
    img = generator(z)

    # For the combined model we will only train the generator
    discriminator.trainable = False

    # The discriminator takes generated images as input and determines validity
    validity = discriminator(img)

    # The combined model  (stacked generator and discriminator)
    # Trains the generator to fool the discriminator
    combined = Model(z, validity)
    combined.compile(loss='binary_crossentropy', optimizer = optimizer)
    
    discriminator.trainable = True 
    return combined

def train(img_shape, X_train, epochs, optimizer, latent_dim, batch_size=128, sample_interval=50):

    channels = img_shape[2] # Número de canales
    
    # Construir y compilar Argquitecturas
    discriminator = build_discriminator(img_shape, optimizer)
    print(discriminator.optimizer.get_config())
    generator = build_generator(img_shape, latent_dim)
    combined = build_gan(generator, discriminator, latent_dim, optimizer)
       
    if channels == 1:
        X_train = np.expand_dims(X_train, axis=3)

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    # Loss output
    g_loss_epochs = np.zeros((epochs, 1))
    d_loss_epochs = np.zeros((epochs, 1))

    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        # Generate a batch of new images
        gen_imgs = generator.predict(noise)

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        # Train the generator (to have the discriminator label samples as valid)
        g_loss = combined.train_on_batch(noise, valid)
        
        #show the final losses
        g_loss_epochs[epoch] = g_loss
        d_loss_epochs[epoch] = d_loss[0]
        
        # Plot the progress
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        # If at save interval => save generated image samples
        if epoch % sample_interval == 0:
            save_model(discriminator, generator, combined)
            sample_images(generator, epoch, img_shape, smp_rows=10, smp_cols=10, save_img=True)
            
    return g_loss_epochs, d_loss_epochs
    
def sample_images(generator, epoch, img_shape, smp_rows=10, smp_cols=10, save_img=True, fig_size=(10, 10)):
    r, c = smp_rows, smp_cols
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise)
    # Rescale images 0 - 255
    gen_imgs = 255*(0.5 * gen_imgs + 0.5)
    # Organiza las matrices de imagenes
    if img_shape[2] == 1:
        gen_imgs = np.uint8(gen_imgs.reshape(smp_rows*smp_cols, img_shape[0], img_shape[1]))
    else:            
        gen_imgs = np.uint8(gen_imgs.reshape(smp_rows*smp_cols, img_shape[0], img_shape[1], 3))
    # Grafica
    plt.figure(figsize=fig_size)
    for i in range(gen_imgs.shape[0]):
        plt.subplot(smp_rows, smp_cols, i+1)
        if img_shape[2] == 1:
            plt.imshow(gen_imgs[i], interpolation='nearest', cmap='gray_r')
        else:
            plt.imshow(gen_imgs[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    if save_img:
        plt.savefig("images/%d.png" % epoch)
    else:
        plt.show()
    plt.close()
    
def save_model(discriminator, generator, combined):
    discriminator.trainable = False
    combined.save('saved_model/combined.h5')
    discriminator.trainable = True
    generator.save('saved_model/generator.h5')
    print(discriminator.optimizer.get_config())
    discriminator.save('saved_model/discriminator.h5')

def plot_gan_losses(g_loss, d_loss):
    plt.plot(g_loss)
    plt.plot(d_loss)
    plt.title('GAN Loss Evaluation')
    plt.ylabel('')
    plt.xlabel('epoch')
    plt.legend(['Generator', 'Discriminator'],loc='upper right')
    plt.savefig("Loss.png")
    plt.show()

if __name__ == '__main__':
    # Cargar datos
    (X_train_m, _) = load_minst_data() 
    
    # Definición de variables
    input_rows = 28 # Tamaño de imagen - filas
    input_cols = 28 # Tamaño de imagen - columnas
    input_channels = 1 # Canales de la imagen
    latent_dim_m = 100 # Espacio latente de acuerdo a cantidad de imagenes a generar durante entrenamiento
    epochs_m = 10000 # Epocas de entrenamiento
    batch_size_m = 100 # Cantidad de imagenes a tomar del X_train para cada ciclo de entrenamiento
    sample_interval_m = 200 # Intervalo de épocas para guardar modelos e imagenes
    img_shape_m = (input_rows, input_cols, input_channels) # Dimensiones de la imagen
    optimizer_m = Adam(0.0002, 0.5) # Optimizador
    
    # Entrenamiento
    g_loss, d_loss = train(img_shape_m, X_train_m, epochs_m, optimizer_m, latent_dim_m, batch_size_m, sample_interval_m)
    
    # Evaluación de pérdidas
    plt.style.use('seaborn-white')
    plot_gan_losses(g_loss, d_loss)