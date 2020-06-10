# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:38:01 2020

@author: nialb
"""
import pandas as pd
import numpy as np
from keras import models
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

from train_helper import train_model

ds_dir = '../augmented_ds'
model_name = 'dense_imnet_full_aug'
metadata = pd.read_csv('../metadata_augmented.csv')

dense_layer = DenseNet121(weights='imagenet', include_top=False, input_shape=(768, 768,3), pooling=None)

# build top model
x = GlobalAveragePooling2D(name = 'avg_pool')(dense_layer.output)
x = Dense(5, activation='sigmoid', name='predictions')(x)

# stitch together
dense_model = models.Model(inputs= dense_layer.input, outputs=x)

# inspect
#dense_model.summary()
dense_model.compile(loss='binary_crossentropy',
                    optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=True),
                    metrics=['accuracy'])

history = train_model(dense_model, model_name, ds_dir, metadata)
