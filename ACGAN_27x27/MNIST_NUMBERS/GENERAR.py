# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 16:50:47 2019

@author: Diego

Basado en:
    https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    http://cican17.com/gan-from-zero-to-hero-part-2-conditional-generation-by-gan/

Labals:
    0: 'tshirt',
    1: 'trouser',
    2: 'pullover',
    3: 'dress',
    4: 'coat',
    5: 'sandal',
    6: 'shirt',
    7: 'sneaker',
    8: 'bag',
    9: 'boot'}
 
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model

#np.random.seed(20) # Semilla random, comentar para resultados siempre diferentes
random_dim = 100

# Genera imagenes de todos los labels
def sample_images(generator, img_shape, etiqueta=1, smp_rows=10, smp_cols=10, save_img=True, fig_size=(10, 10)):
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
        plt.savefig("images_model/%d.png" % etiqueta)
    plt.show()
    plt.close()
    
# Genera imagen individual de acuerdo al label solicitado    
def sample_single_image(generator, img_shape, label=1, etiqueta=1, fig_size=(2, 2)):
    noise = np.random.normal(0, 1, (1*1, 100))
    gen_imgs = generator.predict([noise, np.array(label).reshape((1, ))])   # Genera imagen
    # Rescale images 0 - 255
    gen_imgs = 255*(0.5 * gen_imgs + 0.5)
    # Crea y Guarda figura
    plt.figure(figsize=fig_size)
    if img_shape[2] == 1:
        gen_imgs = np.uint8(gen_imgs.reshape(img_shape[0], img_shape[1]))
        plt.imshow(gen_imgs, interpolation='nearest', cmap='gray_r')
    else:            
        gen_imgs = np.uint8(gen_imgs.reshape(img_shape[0], img_shape[1], 3))
        plt.imshow(gen_imgs, interpolation='nearest')
    plt.axis('off')
    plt.savefig('images_model/imagen_L%d_%d.png' % (label, etiqueta))
    plt.show()
    plt.close()

if __name__ == '__main__':
    input_rows = 28 # Tamaño de imagen - filas
    input_cols = 28 # Tamaño de imagen - columnas
    input_channels = 1 # Canales de la imagen
    img_shape = (input_rows, input_cols, input_channels) # Dimensiones de la imagen  
    
    # load model
    generador = load_model('saved_model/generator.h5') # Cargar el modelo 
    generador.summary() # Imprime características del modelo
    
    # Generar imagenes -- Descomentar la opción deseada
    sample_images(generador, img_shape, etiqueta=1, smp_rows=10, smp_cols=10, save_img=True, fig_size=(10, 10)) # Genera conjunto de imagenes
    #sample_single_image(generador, img_shape, label=10, etiqueta=1) # Genera imagen individual
 
# evaluate loaded model on test data
#score = generador.evaluate(X, Y, verbose=0)