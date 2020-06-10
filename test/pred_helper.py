# -*- coding: utf-8 -*-
"""
Created on Mon May 11 18:49:22 2020

@author: nialb
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import gc
from scipy import interp
import scipy.ndimage as ndimage
from sklearn.metrics import roc_curve, auc, roc_auc_score
from itertools import cycle
from vis.visualization import visualize_saliency, visualize_cam
from vis.utils import utils
from keras import activations

def plot_roc(preds, meta_slice, model_name):
    if(not os.path.exists('../plots')):
        os.makedirs('../plots')
    
    target_cols = ['Normal', 'Pneumonia', 'Virus', 'Bacteria', 'COVID']
    n_classes = len(target_cols)
    true_vals = meta_slice.loc[:, target_cols].to_numpy()
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_vals[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_vals.ravel(), preds.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(target_cols[i], roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plot_title = 'ROC Curve: ' + model_name
    plt.title(plot_title)
    plt.legend(loc="lower right")
    roc_path = "../plots/{}_roc.png".format(model_name)
    plt.savefig(roc_path)
    plt.show()

def save_preds(preds, filenames, meta_slice, save_name):
    if(not os.path.exists('../results')):
        os.makedirs('../results')

    preds_bool = (preds > 0.5)
    
    predictions = preds_bool.astype(int)
    target_cols = ['Normal', 'Pneumonia', 'Virus', 'Bacteria', 'COVID']
    pred_cols = ['Pred_Normal', 'Pred_Pneumonia', 'Pred_Virus', 'Pred_Bacteria', 'Pred_COVID']
    orig_labels = meta_slice.loc[:, target_cols]
    
    results = pd.DataFrame(predictions, columns=pred_cols)
    results['Filenames'] = filenames
    result_cols=['Filenames']+pred_cols
    results=results[result_cols]
    results_final=pd.concat([results, orig_labels], axis=1).reset_index(drop=True)
    
    save_path = '../results/{}_results.csv'.format(save_name)
    results_final.to_csv(save_path, index=False)
    
    return preds_bool

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim and v > 50] += value
    
    final_hsv = cv2.merge((h,s,v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def vis_sal(model, layer_idx, preds_bool, test_slice, test_dir, sal_dir):
    if(not os.path.exists(sal_dir)):
        os.makedirs(sal_dir)
        
    cmap = plt.get_cmap('inferno')
    
    model.layers[layer_idx].activation = activations.linear
    sal_model = utils.apply_modifications(model)
    
    for r_idx in range(0, len(test_slice)):  
        img_name_full = test_slice.iloc[r_idx].loc['X_ray_image_name']
        
        img = cv2.imread(os.path.join(test_dir, img_name_full))
        for c_idx in range(0, 4):
            if preds_bool[r_idx, c_idx] == 1:
                
                grads = visualize_saliency(sal_model, layer_idx, filter_indices=c_idx, 
                                           seed_input=img,
                                           grad_modifier='absolute',
                                           backprop_modifier='guided')
                grads = grads * 7.5
                gaus = ndimage.gaussian_filter(grads, sigma=5)           
                gaus_color = cmap(gaus)
                gaus_color = np.uint8(gaus_color * 255)
                gaus_img = np.delete(gaus_color, 3, 2)
                
                blended_img = cv2.addWeighted(img, 0.9, gaus_img, 1, 0)
                
                img_name_base = img_name_full.split('.jpeg')[0]
                suffix_choices = {0: 'Normal', 1: 'Pneumonia', 2: 'Virus', 3: 'Bacteria', 4: 'COVID' }
                suffix = suffix_choices.get(c_idx, 'default')
                final_img_name = img_name_base + '_' + suffix + '_saliency.jpeg'

                final_img_path = os.path.join(sal_dir, final_img_name)
                cv2.imwrite(final_img_path, blended_img)
        
        if r_idx % 25 == 0 :
            del grads
            del gaus
            del gaus_color
            del gaus_img
            #del hsv
            del blended_img
            gc.collect()


            
def vis_cam(model, pred_idx, conv_idx, backprop, grad, preds_bool, test_slice, test_dir, cam_dir):

    if(not os.path.exists(cam_dir)):
        os.makedirs(cam_dir)
        
    model.layers[pred_idx].activation = activations.linear
    model = utils.apply_modifications(model)
       
    for r_idx in range(0, len(test_slice)):   
        
        img_name_full = test_slice.iloc[r_idx].loc['X_ray_image_name']
                
        image = cv2.imread(os.path.join(test_dir, img_name_full)) 

        for c_idx in range(0, 4):
            if preds_bool[r_idx, c_idx] == 1:
                                      
                cam_output  = visualize_cam(model, pred_idx, c_idx, image,
                                            penultimate_layer_idx = conv_idx,#None,
                                            backprop_modifier     = backprop,
                                            grad_modifier         = grad)
                
                cam_output = np.uint8(cam_output * 255)
                cam_img  = cv2.applyColorMap(cam_output, cv2.COLORMAP_INFERNO)
                
                blended_img = cv2.addWeighted(image, 0.9, cam_img, 0.5, 0)
                img_name_base = img_name_full.split('.jpeg')[0]
                suffix_choices = {0: 'Normal', 1: 'Pneumonia', 2: 'Virus', 3: 'Bacteria', 4: 'COVID' }
                suffix = suffix_choices.get(c_idx, 'default')
                final_img_name = img_name_base + '_' + suffix + '_CAM.jpeg'
                final_img_path = os.path.join(cam_dir, final_img_name)
                cv2.imwrite(final_img_path, blended_img)
                        
        if r_idx % 5 == 0:        
            del cam_output
            del cam_img
            del image
            del blended_img        
            
            gc.collect() 