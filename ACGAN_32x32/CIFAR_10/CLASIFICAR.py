# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 17:14:36 2019

@author: Diego

Basado en:
    https://sophia.ddns.net/deep_learning/gan/MultiGenerator_GAN_Keras.html
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from keras.datasets import cifar10

def load_minst_data():
    # load the data
    #x_train = np.load('Data/Data_Train.npy')
    #y_train = np.load('Data/Data_Labels.npy')
    (_, _), (x_test, y_test) = cifar10.load_data()
    y_test = np.squeeze(y_test); # activar solo para cifar 10 #*****IMPORTANTE*****#
    # normalize our inputs to be in the range[-1, 1] 
    x_test = (x_test.astype(np.float32) - 127.5)/127.5
    # convert x_train with a shape of (60000, 28, 28) to (60000, 784) so we have
    # 784 columns per row
    return (x_test, y_test)


if __name__ == '__main__': 
    
    # Definición de variables
    channels =3 # Canales de la imagen
    
    # Cargar datos para test
    x_test, y_test = load_minst_data() 
    if channels == 1:
        x_test = np.expand_dims(x_test, axis=3)
    
    # Cargar discriminador
    discriminator = load_model('saved_model/discriminator.h5')

    # Evaluar discriminador
    test_res = discriminator.predict(x_test[:1000])
    test_res = np.argmax(test_res[1],1)
    
    # Evaluar capacidad
    correct_prediction = np.equal(test_res, y_test[:1000])
    accuracy = np.mean(correct_prediction.astype(np.float32))
    print("Accuracy: {} %".format(accuracy * 100))
    
    # Ejemplo
    print(y_test[:20])
    print(test_res[:20])