# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 15:10:26 2017

@author: www.deeplearningbrasil.com.br
"""
"""
Bibliotecas utilizadas:
Keras
numpy
opencv (usada apenas para abrira a imagem - pode ser substituida por outra, como numpy ndarray)
matplotlib
"""
from keras.layers import Convolution2D, MaxPooling2D, Activation
from keras.models import Sequential

import numpy as np
import matplotlib.pyplot as plt
#import cv2
import matplotlib.image as mpimg


cat = mpimg.imread('cat.png')
plt.imshow(cat)

cat.shape

model = Sequential()
model.add(Convolution2D(3,3,3,input_shape=cat.shape))

cat_batch = np.expand_dims(cat,axis=0)
conv_cat = model.predict(cat_batch)

def visualize_cat(model, cat):
    cat_batch = np.expand_dims(cat,axis=0)
    conv_cat = model.predict(cat_batch)
    conv_cat = np.squeeze(conv_cat, axis=0)
    print(conv_cat.shape)
    plt.imshow(conv_cat)
    
visualize_cat(model, cat)

model = Sequential()
model.add(Convolution2D(3, (10,10), input_shape=cat.shape))


visualize_cat(model, cat)

def imprime_gatinho_fofo(model, cat):
    cat_batch = np.expand_dims(cat,axis=0)
    conv_cat2 = model.predict(cat_batch)
    conv_cat2 = np.squeeze(conv_cat2, axis=0)
    print(conv_cat2.shape)
    conv_cat2 = conv_cat2.reshape(conv_cat2.shape[:2])
    print(conv_cat2.shape)
    plt.imshow(conv_cat2)
    
model = Sequential()
model.add(Convolution2D(1, (3,3), input_shape=cat.shape))

imprime_gatinho_fofo(model, cat)    

model = Sequential()
model.add(Convolution2D(1, (15,15), input_shape=cat.shape))

imprime_gatinho_fofo(model, cat)


model = Sequential()
model.add(Convolution2D(1, (3,3), input_shape=cat.shape))
model.add(Activation('relu'))

imprime_gatinho_fofo(model, cat)


model = Sequential()
model.add(Convolution2D(3, (3,3), input_shape=cat.shape))
model.add(Activation('relu'))

visualize_cat(model, cat)


model = Sequential()
model.add(Convolution2D(1, (3,3), input_shape=cat.shape))
model.add(MaxPooling2D(pool_size=(5,5)))

imprime_gatinho_fofo(model, cat)

model = Sequential()
model.add(Convolution2D(3, (3,3), input_shape=cat.shape))
model.add(MaxPooling2D(pool_size=(5,5)))

visualize_cat(model, cat)

model = Sequential()
model.add(Convolution2D(1, (3,3), input_shape=cat.shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(5,5)))

imprime_gatinho_fofo(model, cat)

model = Sequential()
model.add(Convolution2D(3, (3,3), input_shape=cat.shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(5,5)))

visualize_cat(model, cat)


model = Sequential()
model.add(Convolution2D(1, (3,3), input_shape=cat.shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Convolution2D(1, (3,3), input_shape=cat.shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

imprime_gatinho_fofo(model, cat)

model = Sequential()
model.add(Convolution2D(3, (3,3), input_shape=cat.shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Convolution2D(1, (3,3), input_shape=cat.shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

imprime_gatinho_fofo(model, cat)

model = Sequential()
model.add(Convolution2D(3, (3,3), input_shape=cat.shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Convolution2D(3, (3,3), input_shape=cat.shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

visualize_cat(model, cat)