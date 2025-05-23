'''
Trains a network to estimate z from Paramecium image.
'''
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import GaussianNoise, Dropout
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from tkinter import filedialog
import tkinter as tk
from gui.gui import *
import yaml
import pandas as pd
from time import time
import numpy as np
import sys
from tensorflow.keras.callbacks import ReduceLROnPlateau
from augmentation.augmentation import *
from models import *
from gui.visualization import *
import tqdm
import zipfile
import io
import random

AUTOTUNE = tf.data.AUTOTUNE

root = tk.Tk()
root.withdraw()

### Folders
path = filedialog.askdirectory(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a dataset folder')
img_path = os.path.join(path, 'images')
if os.path.exists(img_path+'.zip'):
    img_path = img_path+'.zip'
    zipped = True
else:
    zipped = False
label_path = os.path.join(path, 'labels.csv')

### Parameters
parameters = [('epochs', 'Epochs', 500), # number of training epochs
              ('model', 'Model', list(models.keys())),
              ('load_checkpoint', 'Load checkpoint', False), # check this if you want to load a previously trained network
              ('predict_sigma', 'Predict sigma', False), # this exists only for synthetic datasets; ignore in most cases
              ('filename_suffix', 'Filename suffix', ''), # this is a suffix for produced files
              ('background_subtracted', 'Background subtracted', True), # check if background was subtracted
              ('normalize', 'Intensity normalization', False), # normalize by mean intensity, usually not a good idea
              ('max_threshold', 'Maximum threshold', 0), # augmentation with thresholding; normally not used
              ('noise_sigma', 'Noise', 0), # noise augmentation, in pixel intensity (normally 0-255)
              ('min_scaling', 'Minimum intensity scaling', 1.), # intensity scaling augmentation
              ('max_scaling', 'Maximum intensity scaling', 1.), # intensity scaling augmentation
              ('learning_rate', 'Learning rate', 5e-4),
              ('patience', 'Patience', 4), # how many epochs when the error can increase before the learning rate is halved
              ('frozen', 'Frozen', False), # for some complicated models, the base can be frozen
              ('shuffle', 'Shuffle', True), # post-split shuffle # maybe should be automatic, and have optional pre-split shuffle
              ('visualize', 'Visualize only', False) # this is to visualize the dataset
              ]
param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
P = param_dialog.value
save_checkpoint = True
# Suffix
model_name = P['model']
suffix = model_name
if len(P['filename_suffix'])>0:
    suffix += '_'+P['filename_suffix']
checkpoint_filename = os.path.join(path,'best_z_'+suffix+'.tf')
parameter_path = os.path.join(path, 'training_'+suffix+'.yaml')
history_path = os.path.join(path, 'history_'+suffix+'.csv')
model_filename = os.path.join(path, 'model_'+suffix+'.txt')
dataset_parameter_path = os.path.join(path, 'labels.yaml')
batch_size = 128
validation_ratio = 0.2 # proportion of images used for validation

## Read data
df = pd.read_csv(label_path)

## Shuffle
#df = df.sample(frac=1).reset_index(drop=True)

## Read dataset parameters
with open(dataset_parameter_path, 'r') as f:
    P_dataset = yaml.safe_load(f)
# Normalization factor; we ignore for now
#normalization = P_dataset.get('normalization', 1.)
normalization = 1. # perhaps should be .1; also not at loading time
min_threshold = 0.
max_threshold = P['max_threshold']*normalization
noise_sigma = P['noise_sigma']*normalization

## Extract filenames and labels
filenames = df['filename'].values
if P['predict_sigma']:
    if 'sigma' in df:
        labels = df['sigma'].values
    else:
        labels = df['z'].values
else:
    try:
        labels = df['mean_z'].values
    except KeyError:
        labels = df['z'].values
#filenames = [os.path.join(img_path, name) for name in filenames]
n = len(filenames)

## Load images
def load_image(filename):
    # Load the image from the file path
    img = tf.io.read_file(filename)
    img = tf.image.decode_png(img, channels=1)
    img = tf.cast(img, dtype=tf.float32)
    return img

if zipped:
    with zipfile.ZipFile(img_path, 'r') as zip_ref:
        file_data = {name: zip_ref.read(name) for name in zip_ref.namelist()}
        images = [tf.cast(tf.image.decode_png(data, channels=1), dtype=tf.float32)
                  for file_name, data in tqdm.tqdm(file_data.items(), desc='loading dataset')]
else:
    images = [load_image(os.path.join(img_path, name)) for name in tqdm.tqdm(filenames, desc='loading dataset')]

## Get image shape
image = np.array(images[0])
shape = image.shape
print("Image shape:", shape)

## Split the dataset into training and validation, and shuffle each dataset after splitting
images = np.array(images)
n_train = int(n*(1-validation_ratio))
train_indices = list(range(n_train))
val_indices = list(range(n_train, n))
if P['shuffle']:
    random.shuffle(train_indices)
    random.shuffle(val_indices)
train_dataset = tf.data.Dataset.from_tensor_slices((images[train_indices], labels[train_indices]))
val_dataset = tf.data.Dataset.from_tensor_slices((images[val_indices], labels[val_indices]))

## Make training and validation sets
#dataset = tf.data.Dataset.from_tensor_slices((images, labels))
#train_dataset, val_dataset = tf.keras.utils.split_dataset(dataset, right_size=validation_ratio, shuffle=P["shuffle"])

## Data augmentation
## for proper rotation, do it from a 1.42 larger square then crop
#train_dataset = train_dataset.map(lambda x, y: (random_rotate(x), y), num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y), num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.map(lambda x, y: (tf.image.random_flip_up_down(x), y), num_parallel_calls=AUTOTUNE)
if max_threshold>0:
    train_dataset = train_dataset.map(lambda x, y: (random_threshold(x, max_threshold), y), num_parallel_calls=AUTOTUNE)
if noise_sigma>0:
    train_dataset = train_dataset.map(lambda x, y: (add_noise(x,noise_sigma), y), num_parallel_calls=AUTOTUNE)
if P['normalize']:
    train_dataset = train_dataset.map(lambda x, y: (normalize_intensity(x), y), num_parallel_calls=AUTOTUNE)
if (P['min_scaling']!=1.) | (P['max_scaling']!=1.):
    train_dataset = train_dataset.map(lambda x, y: (random_intensity(x, P['min_scaling'], P['max_scaling']), y), num_parallel_calls=AUTOTUNE)

# Validation dataset: just intensity normalization
if P['normalize']:
    val_dataset = val_dataset.map(lambda x, y: (normalize_intensity(x), y), num_parallel_calls=AUTOTUNE)

## Prepare with some shuffling
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

if P['visualize']:
    visualize_dataset(train_dataset)
    exit(0)

## Load weights and model from the checkpoint
if P['load_checkpoint']:
    print('Loading previous model')
    #model.load_weights(checkpoint_filename)
    model = tf.keras.models.load_model(checkpoint_filename)
else:
    if P['frozen']:
        model = models[model_name](shape, trainable=False)
    else:
        model = models[model_name](shape)

model.summary()
with open(model_filename, "w") as f:
    sys.stdout = f  # Redirect output to the file
    model.summary()
    sys.stdout = sys.__stdout__  # Reset to default

## Compile the model
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=P['learning_rate']), #'adam', # default learning_rate .001
              loss='mean_squared_error', metrics=['mae'])

reduce_lr = ReduceLROnPlateau(monitor='loss',
                              factor=0.5,         # Reduce learning rate by half
                              patience=P['patience'], # this must be adapted to the dataset size
                              min_lr=1e-7,        # Minimum learning rate
                              verbose=1)

## Define the ModelCheckpoint callback
checkpoint = ModelCheckpoint(
    checkpoint_filename,         # File path where the best weights will be saved
    monitor='val_loss',      # Metric to monitor for saving the best model
    save_best_only=True,      # Save only the best model weights
    mode='min',              # Mode for monitoring (min for loss, max for accuracy)
    verbose=0,                # Print information about the saving process
    save_weights_only=False
)

if save_checkpoint:
    callbacks = [checkpoint, reduce_lr]
else:
    callbacks = [reduce_lr]

## Save parameters (in case it is interrupted)
with open(parameter_path, 'w') as f:
    yaml.dump(P, f)

## Train
t1 = time()
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=P['epochs'],
    callbacks=callbacks
)
t2 = time()
P['time'] = t2-t1

## Evaluate
loss, mae = model.evaluate(val_dataset)
print(f'Validation loss: {loss}, Validation MAE: {mae}')
P['loss'] = loss
P['mae'] = mae
model.save(os.path.join(path,'z_'+suffix+'.tf'))

## Save training history
df = pd.DataFrame(history.history)
if P['load_checkpoint'] and os.path.exists(history_path):
    df.to_csv(history_path, mode='a', index=False, header=False) # add to existing history
else:
    df.to_csv(history_path)

## Plot
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('mae')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

## Save parameters again
with open(parameter_path, 'w') as f:
    yaml.dump(P, f)
