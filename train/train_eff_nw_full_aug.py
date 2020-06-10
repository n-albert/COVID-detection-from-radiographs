# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 00:15:17 2020

@author: nialb
"""
import pandas as pd

from keras.optimizers import Adam
from train_helper import train_model
from efficient_model import EfficientModel, convolutional_block

ds_dir = '../augmented_ds'
model_name = 'eff_nw_full_aug'
metadata = pd.read_csv('../metadata_augmented.csv')

eff_model = EfficientModel(input_shape=(768, 768, 3), classes = 5)
#print(eff_model.summary())
eff_model.compile(loss='binary_crossentropy',
                    optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=True),
                    metrics=['accuracy'])

history = train_model(eff_model, model_name, ds_dir, metadata)

    
