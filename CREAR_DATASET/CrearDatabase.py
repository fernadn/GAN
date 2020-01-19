# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 16:56:24 2019

@author: Diego

Descripción: 
Este código genera un dataset .npy (imágenes y labels) a partir de un conjunto de imángenes
La carpeta Images contiene las imágenes a convertir
La carpeta Data contiene los dataset generados

Basado en:
    https://machinelearningmastery.com/how-to-load-large-datasets-from-directories-for-deep-learning-with-keras/
"""

# Importar librerias
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Crear constructor de imagenes
datagen = ImageDataGenerator()

# Prepara características del dataset principal
# batch_size corresponde al número de imágenes a convertir
# target_size es el tamaño de salida de la imágenes (pixeles)
train_it = datagen.flow_from_directory('images/training/', class_mode='categorical', batch_size=1098, target_size=(128, 128))

# Prepara características del dataset de validación y test --- No necesario
#val_it = datagen.flow_from_directory('data/validation/', class_mode='binary')
#test_it = datagen.flow_from_directory('data/test/', class_mode='binary')

# Extrae componentes: x, labels
batchX, batchy = train_it.next()
y_aux = np.array(batchy)
etc, labels = np.where(y_aux == 1) # Extrae vector labels

# Imprime características del dataset construido
print(batchX.shape)
print(labels.shape)
for sample in labels[:10]: 
    print(sample)    
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

plt.imshow(np.uint8(batchX[2, :, :, :])) # Ejemplo

# Guardar datasets
np.save('Data/Data_Train.npy',batchX)
np.save('Data/Data_Labels.npy',labels)


