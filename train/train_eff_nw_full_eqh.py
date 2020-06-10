# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 00:15:17 2020

@author: nialb
"""
import pandas as pd

from keras.optimizers import Adam
from train_helper import train_model, train_eqh_model
from efficient_model import EfficientModel, convolutional_block

ds_dir = '../equalized_ds'
model_name = 'eff_nw_full_eqh'
metadata = pd.read_csv('../metadata_eqh.csv')

eff_model = EfficientModel(input_shape=(768, 768, 3), classes = 5)
#print(eff_model.summary())
eff_model.compile(loss='binary_crossentropy',
                    optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=True),
                    metrics=['accuracy'])

history = train_eqh_model(eff_model, model_name, ds_dir, metadata)

    
