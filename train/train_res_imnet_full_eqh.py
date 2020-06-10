# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:38:01 2020

@author: nialb
"""
import pandas as pd
import numpy as np
from keras import models
from keras.applications.resnet import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

from train_helper import train_model, train_eqh_model

ds_dir = '../equalized_ds'
model_name = 'res_imnet_full_eqh'
metadata = pd.read_csv('../metadata_eqh.csv')

resnet_layer = ResNet50(weights="imagenet", include_top=False, input_shape=(768, 768,3), pooling=None)
x = GlobalAveragePooling2D(name = 'avg_pool')(resnet_layer.output)
x = Dense(5, activation='sigmoid', name='predictions')(x)

# stitch together
resnet_model = models.Model(inputs= resnet_layer.input, outputs=x)

# inspect
#resnet_model.summary()
resnet_model.compile(loss='binary_crossentropy',
                    optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=True),
                    metrics=['accuracy'])

history = train_eqh_model(resnet_model, model_name, ds_dir, metadata)
    


    
