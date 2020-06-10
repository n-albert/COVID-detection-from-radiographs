# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:20:35 2020

@author: nialb
"""

from keras import layers
from keras import initializers
from keras.engine.input_layer import Input
from keras.models import Model

#Creates custom convolutional block with shortcut convolutional layer
def convolutional_block(inp, filters, length=2, pool=True, stride=1):
    
    output = inp
    
    #sequential segment of block
    for i in range(length):  
        output = layers.Conv2D(filters=filters,
                                kernel_size=3,
                                strides=stride,
                                padding='same',
                                kernel_initializer=initializers.he_normal(),
                                data_format='channels_last')(output)
        
        output = layers.BatchNormalization()(output)      
        output = layers.Activation('relu')(output)
        
    #parallel shortcut segment
    parallel = layers.Conv2D(filters=filters,
                             kernel_size=1,
                             strides=stride**length,
                             padding='same',
                             kernel_initializer=initializers.he_normal(),
                             data_format='channels_last'
                             )(inp)
    
    parallel = layers.BatchNormalization()(parallel)
    parallel = layers.Activation('relu')(parallel)
    
    output = layers.Lambda(lambda x: (x[0] + x[1]) / 2)([output, parallel])
    
    if pool:
        output = layers.MaxPooling2D(pool_size=3,
                                     strides=2)(output)
    
    return output
    
def EfficientModel(input_shape=(768, 768, 3), width=1, classes = 5):
    inp = Input(input_shape)
    
    output = convolutional_block(inp, filters=16*width, stride=2)
    
    output = convolutional_block(output, filters=32*width)
    output = convolutional_block(output, filters=48*width)
    output = convolutional_block(output, filters=64*width)
    
    output = convolutional_block(output, filters=80*width, pool=False)
    
    
    output = layers.GlobalAveragePooling2D()(output)

    #output layer
    logits = layers.Dense(units=classes, 
                          activation='sigmoid', 
                          name='logits',
                          kernel_initializer=initializers.he_normal()
                          )(output)
    
    model = Model(inputs = inp, 
                  outputs = logits, 
                  name='EffModel')
    
    return model