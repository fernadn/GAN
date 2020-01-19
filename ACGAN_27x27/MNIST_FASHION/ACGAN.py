# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:31:32 2019

@author: Diego
"""
#from __future__ import print_function, division
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Descomentar si se encuentran problemas con tensor y --- depende de la versión tensor flow instalada
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from tensorflow.python.keras.layers.convolutional import UpSampling2D, Conv2D
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.datasets import fashion_mnist

np.random.seed(10)
random_dim = 100

# classes dictionary
label_dict = {0: 'tshirt',
             1: 'trouser',
             2: 'pullover',
             3: 'dress',
             4: 'coat',
             5: 'sandal',
             6: 'shirt',
             7: 'sneaker',
             8: 'bag',
             9: 'boot'}

def load_minst_data():
    # load the data
    #x_train = np.load('Data/Data_Train.npy')
    #y_train = np.load('Data/Data_Labels.npy')
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    y_train = np.squeeze(y_train); # activar solo para cifar 10 #*****IMPORTANTE*****#
    # normalize our inputs to be in the range[-1, 1] 
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    # convert x_train with a shape of (60000, 28, 28) to (60000, 784) so we have
    # 784 columns per row
    return (x_train, y_train)

def build_generator(channels, num_classes, latent_dim):
 
    model = Sequential()  
    
    model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
    model.add(Reshape((7, 7, 128)))
    #7x7x128
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    #14x14x128
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    #28x28x128
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    #28x28x64
    model.add(Conv2D(channels, kernel_size=3, padding='same'))
    model.add(Activation("tanh"))
    #28x28x3
    
    model.summary()
 
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, latent_dim)(label))
 
    model_input = multiply([noise, label_embedding])
    img = model(model_input)
 
    return Model([noise, label], img)

def build_discriminator(num_classes, img_shape, optimizer, losses):
 
    model = Sequential() 
    model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    
    model.summary()
 
    img = Input(shape=img_shape)
 
    # Extract feature representation
    features = model(img)
 
    # Determine validity and label of the image
    validity = Dense(1, activation="sigmoid")(features)
    label = Dense(num_classes, activation="softmax")(features)
    
    # Compilar discriminator
    Model2 = Model(img, [validity, label])
    Model2.compile(loss=losses,
            optimizer=optimizer,
            metrics=['accuracy'])
 
    return Model2

def build_gan(generator, discriminator, latent_dim, optimizer, losses):
    # The generator takes noise and the target label as input
    # and generates the corresponding digit of that label
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,))
    img = generator([noise, label])
 
    # For the combined model we will only train the generator
    discriminator.trainable = False
 
    # The discriminator takes generated image as input and determines validity
    # and the label of that image
    valid, target_label = discriminator(img)
 
    # The combined model  (stacked generator and discriminator)
    # Trains the generator to fool the discriminator
    combined = Model([noise, label], [valid, target_label])
    combined.compile(loss=losses, optimizer=optimizer)
    
    discriminator.trainable = True 
    return combined

def train(num_classes, img_shape, channels, x_train, y_train, epochs, optimizer, losses, batch_size, sample_interval, latent_dim):
   
    # Construir y compilar Argquitecturas
    discriminator = build_discriminator(num_classes, img_shape, optimizer, losses)
    print(discriminator.optimizer.get_config())
    generator = build_generator(channels, num_classes, latent_dim)
    combined = build_gan(generator, discriminator, latent_dim, optimizer, losses)
    
    # Genera dimensión adicional cuando channels = 1 
    if channels == 1:
        x_train = np.expand_dims(x_train, axis=3) 
        
    y_train = y_train.reshape(-1, 1)
    
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
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs = x_train[idx]
        # Sample noise as generator input
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
 
        # The labels of the digits that the generator tries to create an
        # image representation of
        sampled_labels = np.random.randint(0, 10, (batch_size, 1))
 
        # Generate a half batch of new images
        gen_imgs = generator.predict([noise, sampled_labels])
        # Image labels. 0-9 if image is valid or 10 if it is generated (fake)
        img_labels = y_train[idx]

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(imgs, [valid, img_labels])
        d_loss_fake = discriminator.train_on_batch(gen_imgs, [fake, sampled_labels])
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # ---------------------
        #  Train Generator
        # ---------------------
        
        # Train the generator
        g_loss = combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])
 
        #show the final losses
        g_loss_epochs[epoch] = g_loss[0]
        d_loss_epochs[epoch] = d_loss[0]
        
        # If at save interval => save generated image samples
        if (epoch == 0) or ((epoch+1) % sample_interval == 0):
            # Plot the progress
            print ("Epoch: %d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))
            save_model(discriminator, generator, combined)
            sample_images(generator, epoch, img_shape, smp_rows=10, smp_cols=10, save_img=True)
 
    return g_loss_epochs, d_loss_epochs

def sample_images(generator, epoch, img_shape, smp_rows=10, smp_cols=10, save_img=True, fig_size=(10, 10)):
    r, c = smp_rows, smp_cols
    noise = np.random.normal(0, 1, (r * c, 100))
    sampled_labels = np.array([num for _ in range(r) for num in range(c)])
    gen_imgs = generator.predict([noise, sampled_labels])
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
 
#def sample_single_image(noise, label):
#    gen_imgs = self.generator.predict([noise, np.array(label).reshape((1, ))])
#    # Rescale images 0 - 1
#    gen_imgs = 0.5 * gen_imgs + 0.5
#    plt.imshow(gen_imgs[0, :, :, 0], cmap='gray')
 
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
    x_train, y_train = load_minst_data() 
    
    # Definición de variables
    input_rows = 28 # Tamaño de imagen - filas
    input_cols = 28 # Tamaño de imagen - columnas
    input_channels = 1 # Canales de la imagen
    latent_dim = 100 # Espacio latente de acuerdo a cantidad de imagenes a generar durante entrenamiento
    epochs = 102 # Epocas de entrenamiento
    batch_size = 100 # Cantidad de imagenes a tomar del X_train para cada ciclo de entrenamiento
    sample_interval = 100 # Intervalo de épocas para guardar modelos e imagenes
    input_classes = pd.Series(y_train).nunique() # Calcula la cantidad de clases en y_train
    img_shape = (input_rows, input_cols, input_channels) # Dimensiones de la imagen
    optimizer = Adam(0.0002, 0.5) # Optimizador
    losses = ['binary_crossentropy', 'sparse_categorical_crossentropy'] # Pérdidas
    print("x_train shape: {}".format(x_train.shape))
    print("y_train.shape:{}".format(y_train.shape))
    
    # Entrenamiento
    g_loss, d_loss = train(input_classes, img_shape, input_channels, x_train, y_train, epochs, optimizer, losses,
                           batch_size, sample_interval, latent_dim)
    
    # Evaluación de pérdidas
    plt.style.use('seaborn-white')
    plot_gan_losses(g_loss, d_loss)

