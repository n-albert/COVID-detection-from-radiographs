# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 23:13:06 2020

@author: nialb
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

metadata = pd.read_csv('metadata_updated.csv')

normal_slice = metadata[(metadata['Label'] == 'Normal') &
                        (metadata['View'] == 'Front')]
pneumonia_virus_base_slice = metadata[(metadata['Label'] == 'Pnemonia') &
                                      (metadata['View'] == 'Front') &
                                      (metadata['Label_1_Virus_category'] == 'Virus') &
                                      (metadata['Label_2_Virus_category'].isnull())]
pneumonia_virus_covid_slice = metadata[(metadata['Label'] == 'Pnemonia') &
                                      (metadata['View'] == 'Front') &
                                      (metadata['Label_1_Virus_category'] == 'Virus') &
                                      ((metadata['Label_2_Virus_category'] == 'SARS') |
                                      (metadata['Label_2_Virus_category'] == 'COVID-19'))]
pneumonia_bacteria_base_slice = metadata[(metadata['Label'] == 'Pnemonia') &
                                          (metadata['View'] == 'Front') &
                                          (metadata['Label_1_Virus_category'] == 'bacteria') &
                                          (metadata['Label_2_Virus_category'].isnull())]
                                      
pneumonia_virus_covid_slice.loc[:,'Label_2_Virus_category'] = 'COVID'

sample_seed = 42

normal_sampled = normal_slice.sample(n=1400, random_state=sample_seed)
pneumonia_virus_base_sampled = pneumonia_virus_base_slice.sample(n=700, random_state=sample_seed)
pneumonia_bacteria_base_sampled = pneumonia_bacteria_base_slice.sample(n=700, random_state=sample_seed)

split_seed = 718

normal_buffer, normal_test = train_test_split(normal_sampled, test_size = 0.2, random_state=split_seed)
normal_train, normal_val = train_test_split(normal_buffer, test_size = 0.2, random_state=split_seed)

pneumonia_virus_base_buffer, pneumonia_virus_base_test = train_test_split(pneumonia_virus_base_sampled, 
                                                                         test_size = 0.2, 
                                                                         random_state=split_seed)
pneumonia_virus_base_train, pneumonia_virus_base_val = train_test_split(pneumonia_virus_base_buffer, 
                                                                         test_size = 0.2, 
                                                                         random_state=split_seed)

pneumonia_virus_covid_buffer, pneumonia_virus_covid_test = train_test_split(pneumonia_virus_covid_slice, 
                                                                           test_size = 0.2, 
                                                                           random_state=split_seed)
pneumonia_virus_covid_train, pneumonia_virus_covid_val = train_test_split(pneumonia_virus_covid_buffer, 
                                                                           test_size = 0.2, 
                                                                           random_state=split_seed)

pneumonia_bacteria_base_buffer, pneumonia_bacteria_base_test = train_test_split(pneumonia_bacteria_base_sampled, 
                                                                               test_size = 0.2, 
                                                                               random_state=split_seed)
pneumonia_bacteria_base_train, pneumonia_bacteria_base_val = train_test_split(pneumonia_bacteria_base_buffer, 
                                                                               test_size = 0.2, 
                                                                               random_state=split_seed)

normal_train.loc[:,'Dataset_type'] = 'train'
normal_val.loc[:,'Dataset_type'] = 'val'
normal_test.loc[:,'Dataset_type'] = 'test'

pneumonia_virus_base_train.loc[:,'Dataset_type'] = 'train'
pneumonia_virus_base_val.loc[:,'Dataset_type'] = 'val'
pneumonia_virus_base_test.loc[:,'Dataset_type'] = 'test'

pneumonia_virus_covid_train.loc[:,'Dataset_type'] = 'train'
pneumonia_virus_covid_val.loc[:,'Dataset_type'] = 'val'
pneumonia_virus_covid_test.loc[:,'Dataset_type'] = 'test'

pneumonia_bacteria_base_train.loc[:,'Dataset_type'] = 'train'
pneumonia_bacteria_base_val.loc[:,'Dataset_type'] = 'val'
pneumonia_bacteria_base_test.loc[:,'Dataset_type'] = 'test'

metadata_sampled = pd.concat([normal_train, 
                              normal_val,
                              normal_test,
                              pneumonia_virus_base_train,
                              pneumonia_virus_base_val,
                              pneumonia_virus_base_test,
                              pneumonia_virus_covid_train,
                              pneumonia_virus_covid_val,
                              pneumonia_virus_covid_test,
                              pneumonia_bacteria_base_train,
                              pneumonia_bacteria_base_val,
                              pneumonia_bacteria_base_test], axis=0).reset_index(drop=True)

metadata_sampled['Label'] = metadata_sampled['Label'].apply(lambda x: x if x != 'Pnemonia' else 'Pneumonia')

#print("Unique Label values:")
#print(metadata_sampled.Label.unique())

#print("Unique Label_1_Virus_category values:")
#print(metadata_sampled.Label_1_Virus_category.unique())

#print("Unique Label_2_Virus_category values:")
#print(metadata_sampled.Label_2_Virus_category.unique())

add_cols=['Normal','Pneumonia','Virus','Bacteria', 'COVID']
for col in add_cols:
    metadata_sampled[col]=np.nan

for i in range(0, len(metadata_sampled)):
    if metadata_sampled.iloc[i].loc['Label'] == 'Normal':
        metadata_sampled.loc[i, 'Normal'] = 1
    else:
        metadata_sampled.loc[i, 'Normal'] = 0
    
    if metadata_sampled.iloc[i].loc['Label'] == 'Pneumonia':
        metadata_sampled.loc[i, 'Pneumonia'] = 1
    else:
        metadata_sampled.loc[i, 'Pneumonia'] = 0
    
    if metadata_sampled.iloc[i].loc['Label_1_Virus_category'] == 'Virus':
        metadata_sampled.loc[i, 'Virus'] = 1
    else:
        metadata_sampled.loc[i, 'Virus'] = 0
        
    if metadata_sampled.iloc[i].loc['Label_1_Virus_category'] == 'bacteria':
        metadata_sampled.loc[i, 'Bacteria'] = 1
    else:
        metadata_sampled.loc[i, 'Bacteria'] = 0
        
    if metadata_sampled.iloc[i].loc['Label_2_Virus_category'] == 'COVID':
        metadata_sampled.loc[i, 'COVID'] = 1
    else:
        metadata_sampled.loc[i, 'COVID'] = 0

drop_cols=['View', 'Label', 'Label_1_Virus_category', 'Label_2_Virus_category']
metadata_sampled.drop(drop_cols, axis=1, inplace=True) 

metadata_sampled.to_csv('metadata_sampled.csv', index=False)


