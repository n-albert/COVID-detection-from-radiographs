# -*- coding: utf-8 -*-
"""
Created on Thu May  7 17:25:05 2020

@author: nialb
"""
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

def train_model(model, model_name, ds_dir, metadata):
    if(not os.path.exists('../models')):
        os.makedirs('../models')
        
    if(not os.path.exists('../plots')):
        os.makedirs('../plots')
    
    train_slice = metadata[metadata['Dataset_type'] == 'train'].copy().reset_index(drop=True)
    val_slice = metadata[metadata['Dataset_type'] == 'val'].copy().reset_index(drop=True)
    target_cols = ['Normal', 'Pneumonia', 'Virus', 'Bacteria', 'COVID']
    
    train_dir = ds_dir + '/train'
    val_dir = ds_dir + '/val'
    datagen = ImageDataGenerator(rescale=1/255.)
    
    train_generator = datagen.flow_from_dataframe(dataframe=train_slice, directory=train_dir,
                                                 x_col='X_ray_image_name',
                                                 y_col=target_cols,
                                                 target_size=(768,768),
                                                 class_mode='raw',
                                                 batch_size=3,
                                                 shuffle=True,
                                                 seed=1)
    
    val_generator = datagen.flow_from_dataframe(dataframe=val_slice, directory=val_dir,
                                                 x_col='X_ray_image_name',
                                                 y_col=target_cols,
                                                 target_size=(768,768),
                                                 class_mode='raw',
                                                 batch_size=3,
                                                 shuffle=True,
                                                 seed=1)
    
    train_steps = (train_generator.n)//(train_generator.batch_size)
    validation_steps = (val_generator.n)//(val_generator.batch_size)
    
    #weight_path="models/{}_best_weights.hdf5".format(model_name)
    #checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
    #                             save_best_only=True, mode='min', save_weights_only = True)
    
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                       patience=3, 
                                       verbose=1, mode='min', epsilon=0.0001, cooldown=2, min_lr=1e-8)
    early = EarlyStopping(monitor="val_loss", 
                          mode="min", 
                          patience=12) # probably needs to be more patient, but kaggle time is limited
    callbacks_list = [early, reduceLROnPlat]
    
    
    history = model.fit_generator(train_generator,
                                        steps_per_epoch=train_steps, 
                                        epochs=30,
                                        validation_data=val_generator,
                                        validation_steps=validation_steps,
                                        callbacks=callbacks_list)
                                        #use_multiprocessing=False)
                                        
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    acc_path = "../plots/{}_accuracy.png".format(model_name)
    plt.savefig(acc_path)
    plt.show()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    loss_path = "../plots/{}_loss.png".format(model_name)
    plt.savefig(loss_path)
    plt.show()
    
    weight_path="../models/{}_weights.hdf5".format(model_name)
    model.save_weights(weight_path)
    model_path = "../models/{}_model".format(model_name)
    model_json = model.to_json()
    with open(model_path,"w") as json_file:
        json_file.write(model_json)
                                        
    return history

def train_eqh_model(model, model_name, ds_dir, metadata):
    if(not os.path.exists('../models')):
        os.makedirs('../models')
        
    if(not os.path.exists('../plots')):
        os.makedirs('../plots')
    
    train_slice = metadata[metadata['Dataset_type'] == 'train'].copy().reset_index(drop=True)
    val_slice = metadata[metadata['Dataset_type'] == 'val'].copy().reset_index(drop=True)
    target_cols = ['Normal', 'Pneumonia', 'Virus', 'Bacteria', 'COVID']
    
    train_dir = ds_dir + '/train'
    val_dir = ds_dir + '/val'
    train_datagen = ImageDataGenerator(rescale=1/255.,
                                 rotation_range=10,
                                 height_shift_range=0.05,
                                 width_shift_range=0.05,
                                 shear_range=0.02,
                                 horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1/255.)
    
    train_generator = train_datagen.flow_from_dataframe(dataframe=train_slice, directory=train_dir,
                                                 x_col='X_ray_image_name',
                                                 y_col=target_cols,
                                                 target_size=(768,768),
                                                 class_mode='raw',
                                                 batch_size=5,
                                                 shuffle=True,
                                                 seed=1)
    
    val_generator = val_datagen.flow_from_dataframe(dataframe=val_slice, directory=val_dir,
                                                 x_col='X_ray_image_name',
                                                 y_col=target_cols,
                                                 target_size=(768,768),
                                                 class_mode='raw',
                                                 batch_size=5,
                                                 shuffle=True,
                                                 seed=1)
    
    train_steps = (2*train_generator.n)//(train_generator.batch_size)
    validation_steps = (2*val_generator.n)//(val_generator.batch_size)
    
    #checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
    #                             save_best_only=True, mode='min', save_weights_only = True)
    
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                       patience=5, 
                                       verbose=1, mode='min', epsilon=0.0001, cooldown=2, min_lr=1e-8)
    early = EarlyStopping(monitor="val_loss", 
                          mode="min", 
                          patience=15) # probably needs to be more patient, but kaggle time is limited
    callbacks_list = [early, reduceLROnPlat]
    
    
    history = model.fit_generator(train_generator,
                                        steps_per_epoch=train_steps, 
                                        epochs=50,
                                        validation_data=val_generator,
                                        validation_steps=validation_steps,
                                        callbacks=callbacks_list)
                                        #use_multiprocessing=False)
                                        
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    acc_path = "../plots/{}_accuracy.png".format(model_name)
    plt.savefig(acc_path)
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    loss_path = "../plots/{}_loss.png".format(model_name)
    plt.savefig(loss_path)
    plt.show()
    
    weight_path="../models/{}_weights.hdf5".format(model_name)
    model.save_weights(weight_path)
    model_path = "../models/{}_model".format(model_name)
    model_json = model.to_json()
    with open(model_path,"w") as json_file:
        json_file.write(model_json)
                                        
    return history