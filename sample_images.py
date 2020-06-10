# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:07:29 2020

@author: nialb
"""

import os
import shutil
import numpy as np
import pandas as pd
import cv2

finalWidth = 768
finalHeight = 768

sample_dir = 'sampled_ds'
sample_train_dir = sample_dir + '\\train'
sample_val_dir = sample_dir + '\\val'
sample_test_dir = sample_dir + '\\test'

if(os.path.exists(sample_dir)):
    shutil.rmtree(sample_dir)
    
os.makedirs(sample_dir)
os.makedirs(sample_train_dir)
os.makedirs(sample_val_dir)
os.makedirs(sample_test_dir)

metadata_sampled = pd.read_csv('metadata_sampled.csv')
metadata_sampled['sampled_file_path'] = ''

for i in range(0, len(metadata_sampled)):
    src_img_path  = metadata_sampled.loc[i, 'orig_file_path']
    src_img_name  = metadata_sampled.loc[i, 'X_ray_image_name']
    ds_type       = metadata_sampled.loc[i, 'Dataset_type']
    
    src_img_data = cv2.imread(src_img_path)   
    img_resized = cv2.resize(src_img_data, (finalWidth, finalHeight))
    
    if ds_type == 'train':
        final_img_path = os.path.join(sample_train_dir, src_img_name)
        metadata_sampled.loc[i, 'sampled_file_path'] = final_img_path
    elif ds_type == 'val':
        final_img_path = os.path.join(sample_val_dir, src_img_name)
        metadata_sampled.loc[i, 'sampled_file_path'] = final_img_path    
    else:
        final_img_path = os.path.join(sample_test_dir, src_img_name)
        metadata_sampled.loc[i, 'sampled_file_path'] = final_img_path
       
        
    cv2.imwrite(final_img_path, img_resized)

metadata_sampled.drop('orig_file_path', axis=1, inplace=True)
metadata_sampled.to_csv('metadata_sampled.csv', index=False)
    
    