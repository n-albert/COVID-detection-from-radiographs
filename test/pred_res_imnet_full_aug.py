# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:58:44 2020

@author: nialb
"""
import pandas as pd
from keras.models import Model, model_from_json
from keras.preprocessing.image import ImageDataGenerator
from vis.utils import utils

from pred_helper import save_preds, vis_cam, plot_roc

ds_dir = '../augmented_ds'
test_dir = ds_dir + '/test'
val_dir = ds_dir + '/val'
vis_dir = '../visualizations'
cam_dir = vis_dir + '/res_imnet_full_aug_cam'

metadata = pd.read_csv('metadata_augmented.csv')
val_slice = metadata[metadata['Dataset_type'] == 'val'].copy().reset_index(drop=True)
test_slice = metadata[metadata['Dataset_type'] == 'test'].copy().reset_index(drop=True)

json_file = open('../models/res_imnet_full_aug_model.json', 'r')
loaded_model = json_file.read()
json_file.close()
model = model_from_json(loaded_model)

model.load_weights('../models/res_imnet_full_aug_weights.hdf5')
model.summary()

datagen = ImageDataGenerator(rescale=1/255.)

val_generator = datagen.flow_from_dataframe(dataframe=val_slice, directory=val_dir,
                                             x_col='X_ray_image_name',
                                             #y_col=target_cols,
                                             target_size=(768,768),
                                             class_mode=None,
                                             batch_size=1,
                                             shuffle=False,
                                             seed=1)

val_steps = (val_generator.n)//(val_generator.batch_size) 
val_preds = model.predict_generator(val_generator,
                                    steps=val_steps,
                                    verbose=1)

plot_roc(val_preds, val_slice, 'res_imnet_full_aug')

test_generator = datagen.flow_from_dataframe(dataframe=test_slice, directory=test_dir,
                                             x_col='X_ray_image_name',
                                             #y_col=target_cols,
                                             target_size=(768,768),
                                             class_mode=None,
                                             batch_size=1,
                                             shuffle=False,
                                             seed=1)
test_filenames = test_generator.filenames
test_steps = (test_generator.n)//(test_generator.batch_size) 

test_generator.reset()
preds = model.predict_generator(test_generator,
                                steps=test_steps,
                                verbose=1)

preds_bool = save_preds(preds, test_filenames, test_slice, 'res_imnet_full_aug')
        
pred_idx = utils.find_layer_idx(model, 'predictions')
conv_idx = utils.find_layer_idx(model, 'conv5_block3_add')
 
vis_cam(model, pred_idx, conv_idx, None, 'relu', preds_bool, test_slice, test_dir, cam_dir) 