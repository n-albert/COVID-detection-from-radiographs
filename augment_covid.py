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

aug_dir = 'augmented_ds'
aug_train_dir = aug_dir + '\\train'
aug_val_dir = aug_dir + '\\val'
aug_test_dir = aug_dir + '\\test'

if(os.path.exists(aug_dir)):
    shutil.rmtree(aug_dir)
    
os.makedirs(aug_dir)
os.makedirs(aug_train_dir)
os.makedirs(aug_val_dir)
os.makedirs(aug_test_dir)

metadata_base = pd.read_csv('metadata_sampled.csv')

#create ImageDataGenerator for augmenting images
train_datagen = ImageDataGenerator(height_shift_range=0.05,
                             rotation_range=5,
                             horizontal_flip=True,
                             brightness_range=[0.9,1.1],
                             zoom_range=[0.9,1.1])

test_datagen = ImageDataGenerator(height_shift_range=0.05,
                             rotation_range=2.5,
                             horizontal_flip=True,
                             brightness_range=[0.95,1.05],
                             zoom_range=[0.95,1.05])

covid_base_slice = metadata_base[metadata_base['COVID'] == 1].copy()
noncovid_slice = metadata_base[metadata_base['COVID'] == 0].copy().reset_index(drop=True)
covid_base_train_slice = covid_base_slice[covid_base_slice['Dataset_type'] == 'train'].copy()
covid_base_val_slice = covid_base_slice[covid_base_slice['Dataset_type'] == 'val'].copy()
covid_base_test_slice = covid_base_slice[covid_base_slice['Dataset_type'] == 'test'].copy()

def create_aug_slice(base_slice, datagen, ds_type, total, savedir, seed):    
    target_cols = ['Normal', 'Pneumonia', 'Virus', 'Bacteria', 'COVID']
    
    tcount = 0    
    for batch in datagen.flow_from_dataframe(
            dataframe=base_slice,
            x_col='sampled_file_path',
            y_col=target_cols,
            target_size=(768,768),
            class_mode='multi_output',
            batch_size=len(base_slice),
            shuffle=False,
            seed=seed,
            save_to_dir= savedir,
            save_prefix='aug',
            save_format='jpeg'):
        
        tcount += 1
        if tcount == total:
            break
    
    new_cols = ['X_ray_image_name', 'Dataset_type', 'augmented_file_path'] + target_cols
    
    augmented_slice = pd.DataFrame(columns=new_cols)
    
    for root, dirs, files in os.walk(savedir):
        for name in files:
            filepath = os.path.join(root, name)
            augmented_slice = augmented_slice.append(pd.Series([name, ds_type, filepath, 0, 1, 1, 0, 1], index=augmented_slice.columns), ignore_index=True)
    
    return augmented_slice

covid_aug_train_slice = create_aug_slice(covid_base_train_slice, train_datagen, 'train', 15, aug_train_dir, 1)
covid_aug_val_slice = create_aug_slice(covid_base_val_slice, train_datagen, 'val', 15, aug_val_dir, 1)
covid_aug_test_slice = create_aug_slice(covid_base_test_slice, test_datagen, 'test', 12, aug_test_dir, 1)

for i in range(0, len(noncovid_slice)):
    src_img_path  = noncovid_slice.loc[i, 'sampled_file_path']
    src_img_name  = noncovid_slice.loc[i, 'X_ray_image_name']
    ds_type       = noncovid_slice.loc[i, 'Dataset_type']
    
    src_img_data = cv2.imread(src_img_path)   
    
    if ds_type == 'train':
        final_img_path = os.path.join(aug_train_dir, src_img_name)
        noncovid_slice.loc[i, 'augmented_file_path'] = final_img_path
        noncovid_slice.loc[i, 'Dataset_type'] = 'train'
    elif ds_type == 'val':
        final_img_path = os.path.join(aug_val_dir, src_img_name)
        noncovid_slice.loc[i, 'augmented_file_path'] = final_img_path  
        noncovid_slice.loc[i, 'Dataset_type'] = 'val'
    else:
        final_img_path = os.path.join(aug_test_dir, src_img_name)
        noncovid_slice.loc[i, 'augmented_file_path'] = final_img_path
        noncovid_slice.loc[i, 'Dataset_type'] = 'test'
           
    cv2.imwrite(final_img_path, src_img_data)
    
metadata_augmented = pd.concat([noncovid_slice, 
                                covid_aug_train_slice, 
                                covid_aug_val_slice, 
                                covid_aug_test_slice], join='inner', axis=0, ignore_index=True).reset_index(drop=True)

metadata_augmented.to_csv('metadata_augmented.csv', index=False)