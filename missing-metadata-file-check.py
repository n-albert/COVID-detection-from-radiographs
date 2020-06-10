# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 20:15:04 2020

@author: nialb
"""

import os
import numpy as np
import pandas as pd

ds_dir = 'Coronahack-Chest-XRay-Dataset'

metadata_names = []
file_names = []

md_names_no_file = []
file_names_no_md = []

metadata = pd.read_csv('Chest_xray_Corona_Metadata.csv')

metadata.drop(metadata.columns[0], axis = 1, inplace = True)

metadata_names = metadata['X_ray_image_name'].values

for root, dirs, files in os.walk(ds_dir):
    for name in files:
        file_names.append(name)
        metadata.loc[metadata['X_ray_image_name'] == name, 'orig_file_path'] = os.path.join(root, name)
        
for metadata_name in metadata_names:
    if metadata_name not in file_names:
        md_names_no_file.append(metadata_name)
        
for file_name in file_names:
    if file_name not in metadata_names:
        file_names_no_md.append(file_name)
        
missing_ds = pd.DataFrame()
missing_ds['metadata_without_file_names'] = md_names_no_file
missing_ds['files_without_metadata_names'] = file_names_no_md

missing_ds.to_csv('missing.csv', index=False)
metadata.to_csv('metadata_updated.csv', index=False)

import seaborn as sns
import matplotlib.pyplot as plt

plots= pd.DataFrame(columns=[])
fig, ax =plt.subplots(1,3, figsize=(15, 7.5))
sns.countplot(x="Label", hue="Dataset_type", data=metadata, ax=ax[0])
sns.countplot(x="Label_1_Virus_category", hue="Dataset_type", data=metadata, ax=ax[1])
sns.countplot(x="Label_2_Virus_category", hue="Dataset_type", data=metadata, ax=ax[2])
fig.show()