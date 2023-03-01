import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from conf.settings import Settings

settings = Settings()

model_folder = settings.model_folder
metrics_folder = settings.metrics_folder
patience = settings.patience


def learning_scheduler():
    def scheduler(epoch, lr):
        if epoch % 10 == 0:
            return lr * 0.1   
        return lr

    callback_lr_decay = tf.keras.callbacks.LearningRateScheduler(scheduler)
    return callback_lr_decay

def callbacks():    
    
    checkpoint_file = model_folder + "/vae_weights.h5"
    checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_loss', verbose=0, save_best_only=True,
                                 save_weights_only=True, mode='auto')

    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')
    lr_decay =  learning_scheduler()
    return[checkpoint,early,lr_decay]   
    
   

    
def plot_latent_space(pictures,classes,estimator):
    
    class_dict = {
    0:"T-shirt/top",
    1:"Trouser",
    2:"Pullover",
    3:"Dress",
    4:"Coat",
    5:"Sandal",
    6:"Shirt",
    7:"Sneaker",
    8:"Bag",
    9:"Ankle boot",
    }
    
    complete_latent_space = estimator.encoder.predict(pictures)
    
    graphDf = pd.DataFrame(data = complete_latent_space , columns = ['z1', 'z2','z3'])
    graphDf = pd.concat([graphDf, pd.DataFrame(classes)], axis = 1)
    graphDf.rename(columns = {0 : "class"},inplace = True)  
    graphDf["class"].replace(to_replace = class_dict, inplace = True)
    
    fig = px.scatter_3d(graphDf, x='z1', y='z2', z='z3',color="class")
    fig.write_html(metrics_folder + '/LatentSpace.html', auto_open=True)
    
    

def morph(pictures,classes,class_a,class_b,estimator,amount_of_vector_multipliers):
    
    latent_space_a = estimator.encoder.predict(pictures[classes == class_a])
    mean_latent_space_a= np.expand_dims(latent_space_a.mean(axis = 0),axis = 0)

    latent_space_b = estimator.encoder.predict(pictures[classes == class_b])
    mean_latent_space_b = np.expand_dims(latent_space_b.mean(axis = 0),axis = 0)

    class_ab_dict  = {}
    for i in np.linspace(0, 2, num = amount_of_vector_multipliers, endpoint=True, retstep=False, dtype=None, axis=0):   
        class_ab_dict[i] = estimator.decoder.predict( mean_latent_space_a + i *  mean_latent_space_b)
    
    keys = list(class_ab_dict.keys())
    current_key = 0
    fig, axs = plt.subplots(1,amount_of_vector_multipliers, figsize=(30,30))
    for col in range(amount_of_vector_multipliers):      
        axs[col].imshow(class_ab_dict[keys[current_key]][0],cmap = "gray")
        axs[col].set_title(f'alpha:{round(keys[current_key],2)}')
        current_key += 1   