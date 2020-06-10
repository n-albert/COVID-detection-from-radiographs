# -*- coding: utf-8 -*-
"""
Created on Sat May 23 22:28:48 2020

@author: nialb
"""
import pandas as pd
import numpy as np
import os
from sklearn.metrics import hamming_loss, accuracy_score, classification_report

results_dir = 'results'
metrics_dir = 'metrics'

if(not os.path.exists(metrics_dir)):
    os.makedirs(metrics_dir)

for root, dirs, files in os.walk(results_dir):
    for name in files:
        
        base_name = name.split('_results')[0]
        results = pd.read_csv(os.path.join(root, name))
        
        true_cols = ['Normal', 'Pneumonia', 'Virus', 'Bacteria', 'COVID']
        pred_cols = ['Pred_Normal', 'Pred_Pneumonia', 'Pred_Virus', 'Pred_Bacteria', 'Pred_COVID']
        
        true_slice = results.loc[:, true_cols]#.to_numpy().astype('int32')
        pred_slice = results.loc[:, pred_cols]#.to_numpy().astype('int32')
        
        print('{} Classification Report'.format(base_name))
        report = classification_report(true_slice,pred_slice, target_names=true_cols, output_dict=True)
        
        print("Accuracy Score:", accuracy_score(true_slice, pred_slice))
        print("Hamming Loss:", hamming_loss(true_slice, pred_slice))
        
        metrics = pd.DataFrame(report).transpose()
        metrics.loc['-'] = '-'
        metrics.loc['Accuracy_Score'] = accuracy_score(true_slice, pred_slice)
        metrics.loc['Hamming_Loss'] = hamming_loss(true_slice, pred_slice)
        print('')
        
        save_name = '{}_metrics.csv'.format(base_name)
        
        metrics.to_csv(os.path.join(metrics_dir, save_name), index=True)

        
