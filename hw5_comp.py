#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 00:22:16 2020

@author: jahnvirc
"""

import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers, optimizers
import numpy as np


(xtr, ytr), (xte, yte) = cifar10.load_data()
xtr = xtr.astype('float32')
xte = xte.astype('float32')

mean1 = np.mean(xtr,axis=(0,1,2,3))
std1 = np.std(xtr,axis=(0,1,2,3))
xtr = (xtr-mean1)/(std1+0.0000001)
xte = (xte-mean1)/(std1+0.0000001)

cla_cnt = 10
ytr = np_utils.to_categorical(ytr,cla_cnt)
yte = np_utils.to_categorical(yte,cla_cnt)

conv_val = 32
ld_val = 0.0001
pd_val = 'same'
act_val = 'elu'
conv_mod = Sequential()
def cnn_conv(pd_val,act_val,ld_val,conv_val,i,dr_val):
    conv_mod.add(Conv2D(conv_val*i, (3,3), padding=pd_val, kernel_regularizer=regularizers.l2(ld_val), input_shape=xtr.shape[1:]))
    conv_mod.add(Activation(act_val))
    conv_mod.add(BatchNormalization())
    conv_mod.add(Conv2D(conv_val, (3,3), padding=pd_val, kernel_regularizer=regularizers.l2(ld_val)))
    conv_mod.add(Activation(act_val))
    conv_mod.add(BatchNormalization())
    conv_mod.add(MaxPooling2D(pool_size=(2,2)))
    conv_mod.add(Dropout(dr_val))
    
cnn_conv(pd_val,act_val,ld_val,conv_val,1,0.2)
cnn_conv(pd_val,act_val,ld_val,conv_val,2,0.3)
cnn_conv(pd_val,act_val,ld_val,conv_val,4,0.4)
conv_mod.add(Flatten())
conv_mod.add(Dense(10, activation='softmax'))
conv_mod.summary()

dat_aug = ImageDataGenerator(rotation_range=15,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)
dat_aug.fit(xtr)

bt_sz = 64
epch= 80
conv_mod5=[]
def fit_dat(bt_sz,epch,i,lr_val):
    fit_val = keras.optimizers.rmsprop(lr=0.001,decay=0.000001)
    conv_mod.compile(loss='categorical_crossentropy',optimizer=fit_val,metrics=['accuracy'])
    conv_mod5=conv_mod.fit_generator(dat_aug.flow(xtr, ytr, batch_size=bt_sz),steps_per_epoch=xtr.shape[0]//bt_sz,epochs=epch*i,verbose=1,validation_data=(xte,yte))

fit_dat(bt_sz,epch,3,0.001)
fit_dat(bt_sz,epch,1,0.0005)
fit_dat(bt_sz,epch,1,0.0003)

output = conv_mod.evaluate(xte, yte, batch_size=128, verbose=1)
print('Test loss:', output[0])
print('Test accuracy:', output[1])
print('\nTest result: %.3f loss: %.3f' % (output[1]*100,output[0]))