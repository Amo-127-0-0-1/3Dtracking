'''
identical to predict_movie.py but with timing for benchmarking.

Predict z on a movie (tiff folder).
If it is a .tiff folder, select one of the tiff files.
If it is a zipped folder, select the zip file.
'''
import numpy as np
import os
import time
from tkinter import filedialog
from gui.gui import *
import yaml
import tkinter as tk
from tracking.load_tracking import *
import tqdm
from tensorflow.keras.models import load_model
from movie.movie import *
from movie.cell_extraction import *

# Timing setup
timers = {}


def tic(label):
    timers[label] = time.time()


def toc(label):
    timers[label] = time.time() - timers[label]


root = tk.Tk()
root.withdraw()

### Files and folders
movie_filename = filedialog.askopenfilename(initialdir=os.path.expanduser('~/Downloads/'), title='Choose a movie file')
traj_filename = filedialog.askopenfilename(initialdir=os.path.dirname(movie_filename), title='Choose a trajectory file')
name, ext = os.path.splitext(traj_filename)
output_trajectories = name + '_with_z.csv'
model_filename = filedialog.askdirectory(initialdir=os.path.dirname(movie_filename), title='Choose a model')

## Parameters
parameters = [('in_pixel', 'Trajectory in pixel', True),
              ('pixel_size', 'Pixel size (um)', 5.06),
              ('normalize', 'Intensity normalization', False)]
param_dialog = ParametersDialog(title='Enter parameters', parameters=parameters)
P = param_dialog.value
pixel_size = P['pixel_size']


tic('total')
### Load the trained model
tic('model_loading')
model = load_model(model_filename,
                   custom_objects={'modified_mae': None, 'mean_abs_difference_metric': None, 'combined_loss': None})
image_size = model.input_shape[1]
half_img_size = image_size // 2
toc('model_loading')

### Load trajectories
tic('trajectory_loading')
data = magic_load_trajectories(traj_filename)
data['z'] = np.nan
toc('trajectory_loading')

### Put data in pixels
if P['in_pixel']:
    pixel_size = 1.

### Open movie
tic('movie_loading')
if movie_filename.endswith('.zip'):
    movie = MovieZip(movie_filename, auto_invert=True, gray=True)
else:
    image_path = os.path.dirname(movie_filename)
    movie = MovieFolder(image_path, auto_invert=True)
toc('movie_loading')

### Get image size
image = movie.current_frame()
width, height = image.shape[1], image.shape[0]
n_frames = data['frame'].nunique()

### Iterate through frames
tic('total_prediction_loop')
prediction_times = []
previous_position = 0
for image in tqdm.tqdm(movie.frames(), total=n_frames):
    data_frame = data[data['frame'] == previous_position]
    if len(data_frame) > 0:
        snippets = extract_cells(image, data_frame, image_size, pixel_size=pixel_size)

        frame_label = f'frame_{previous_position}_prediction'
        tic(frame_label)
        if P['normalize']:
            predictions = model.predict(np.array([snippet / (np.mean(snippet) + 1e-8) for snippet in snippets]))
        else:
            predictions = model.predict(np.array(snippets))
        toc(frame_label)
        prediction_times.append((frame_label, timers[frame_label]))

        data.loc[data['frame'] == previous_position, 'z'] = predictions
    previous_position = movie.position
toc('total_prediction_loop')

print(data.head())

### Saving results
tic('save_csv')
data.to_csv(output_trajectories)
toc('save_csv')
toc('total')

# Timing Summary
print("\n=== Timing Summary ===")
print(f"Model loading: {timers['model_loading']:.2f} sec")
print(f"Trajectory loading: {timers['trajectory_loading']:.2f} sec")
print(f"Movie loading: {timers['movie_loading']:.2f} sec")
print(f"Total prediction loop: {timers['total_prediction_loop']:.2f} sec")
print(f"CSV saving: {timers['save_csv']:.2f} sec")
print(f"Total time:      {timers['total']:.2f} sec")



# 11798
# === Timing Summary ===
# Model loading: 1.15 sec
# Trajectory loading: 0.15 sec
# Movie loading: 0.12 sec
# Total prediction loop: 594.17 sec
# CSV saving: 1.03 sec
# Total time:      596.63 sec