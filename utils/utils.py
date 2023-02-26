import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

from conf.settings import Settings

settings = Settings()

model_folder = settings.model_folder
img_height = settings.img_height
img_width = settings.img_width

def show_pictures(pictures,classes):
    size = pictures.shape[0]
    fig, axs = plt.subplots(5, 5, figsize=(12,8))
    for row in range(5):
        for col in range(5):
            img_idx = np.random.randint(size - 1)       
            axs[row, col].imshow(pictures[img_idx],cmap = "gray")
            axs[row, col].axis('off')
            axs[row, col].set_title(f'Class: {classes[img_idx]}')
            
def create_folder(path):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)   

        
def process_data(x_train, y_train,x_test, y_test):
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = x_train.reshape(x_train.shape[0], img_height, img_width, 1)
    x_test = x_test.reshape(x_test.shape[0], img_height, img_width, 1)
    
    return x_train, y_train,x_test, y_test

def show_reconstructed(pictures,estimator):
    size = pictures.shape[0]
    fig, axs = plt.subplots(4, 4, figsize=(12,8))
    for row in range(4):
        for col in range(0,3,2):       
            img_idx = np.random.randint(size - 1)    
            axs[row, col].imshow(pictures[img_idx],cmap = "gray")

            latent = estimator.encoder.predict(np.expand_dims(pictures[img_idx],axis = 0))
            reconstruction = estimator.decoder.predict(latent)        
            axs[row, col+1].imshow(reconstruction[0],cmap = "gray")   

            axs[row, col].axis('off')
            axs[row, col].set_title(f'Original')
            axs[row, col+1].axis('off') 
            axs[row, col+1].set_title(f'Reconstructed')  
            

            
