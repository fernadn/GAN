# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 16:07:19 2019

@author: Diego
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def load_minst_data():
    # load the data
    x_train = np.load('Data/Data_Train.npy')
    y_train = np.load('Data/Data_Labels.npy')
    # normalize our inputs to be in the range[-1, 1] 
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    return (x_train, y_train)

# Cargar dataset
x_train, y_train = load_minst_data()

# Características dataset
print('Batch shape=%s, min=%.3f, max=%.3f' % (x_train.shape, x_train.min(), x_train.max()))
print('Labels shape=%d' % y_train.shape)

# Primeros 10 labels
print("Ejemplos Labels: ")
for sample in y_train[:10]: 
    print(sample)
    
# Imagen de ejemplo
plt.imshow(np.uint8(255*(0.5*x_train[2, :, :, :]+0.5))) # Ejemplo
plt.axis('off')