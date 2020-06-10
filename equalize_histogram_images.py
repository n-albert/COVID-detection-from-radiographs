# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 23:02:52 2020

@author: nialb
"""
import os
import shutil

import numpy as np
import pandas as pd

import cv2
from keras.preprocessing.image import ImageDataGenerator

eqh_dir = 'equalized_ds'
eqh_train_dir = eqh_dir + '\\train'
eqh_val_dir = eqh_dir + '\\val'
eqh_test_dir = eqh_dir + '\\test'

if(os.path.exists(eqh_dir)):
    shutil.rmtree(eqh_dir)
    
os.makedirs(eqh_dir)
os.makedirs(eqh_train_dir)
os.makedirs(eqh_val_dir)
os.makedirs(eqh_test_dir)

def clahe_prep(img):    
    img_crp = img[40:738, 40:738]
    img_sized = cv2.resize(img_crp, (768,768))
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
    lab = cv2.cvtColor(img_sized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))

    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return np.uint8(img)

metadata_base = pd.read_csv('metadata_augmented.csv')
metadata_base['eqh_file_path'] = ''

for i in range(0, len(metadata_base)):
    file_name = metadata_base.iloc[i].loc['X_ray_image_name']
    base_path = metadata_base.iloc[i].loc['augmented_file_path']
    ds_type = metadata_base.iloc[i].loc['Dataset_type']
    base_img = cv2.imread(base_path)
    enh_img = clahe_prep(base_img)
    
    if ds_type == 'train':
        save_path = os.path.join(eqh_train_dir, file_name)
    elif ds_type == 'val': 
        save_path = os.path.join(eqh_val_dir, file_name)
    elif ds_type == 'test': 
        save_path = os.path.join(eqh_test_dir, file_name)
    else:
        save_path = os.path.join(eqh_dir, file_name)
    
    metadata_base['eqh_file_path'].iloc[i] = save_path
    cv2.imwrite(save_path, enh_img)
    
metadata_eqh = metadata_base.drop(['augmented_file_path'], axis=1)
metadata_eqh.to_csv('metadata_eqh.csv', index=False)
