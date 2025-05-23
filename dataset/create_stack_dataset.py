'''
Makes a dataset from several tiff folders.
Each folder corresponds to a particular z.

Assumed:
- subfolders are sorted by name, in order of depth
- an `images` folder inside each subfolder
- trajectory file inside each folder (txt/csv/tsv), with a `z` column.
'''
import numpy as np
import os
import pandas as pd
from tkinter import filedialog
from gui.gui import *
import yaml
import tkinter as tk
from tracking.load_tracking import *
import imageio
import tqdm
from movie.movie import *
from movie.cell_extraction import *
import zipfile
import io

root = tk.Tk()
root.withdraw()

### Files and folders
stack_folder = filedialog.askdirectory(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a folder')
path = filedialog.askdirectory(initialdir=stack_folder, message='Choose a dataset folder')

img_path = os.path.join(path, 'images')
label_path = os.path.join(path, 'labels.csv')
parameter_path = os.path.join(path, 'labels.yaml')

### Parameters
parameters = [('step', 'Step', -100.), # in units of the trajectory files (pixel or um)
              ('pixel_size', 'Pixel size (um)', 5.),
              ('image_size', 'Image size (um)', 200),
              ('zip', 'Zip', True)
              ]
param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
P = param_dialog.value

pixel_size = P['pixel_size']
P['image_size'] = (int(P['image_size']/pixel_size)//32)*32
image_size = P['image_size']
half_img_size = int(image_size/2)

if (not P['zip']) & (not os.path.exists(img_path)):
    os.mkdir(img_path)

### Data frame
df = pd.DataFrame(columns=['filename', 'mean_z'])

### Get folders
folders = [os.path.join(stack_folder, d) for d in os.listdir(stack_folder)
           if os.path.isdir(os.path.join(stack_folder, d)) and
           not (os.path.join(stack_folder, d) == path)]
folders.sort()

### Iterate folders
if P['zip']:
    zip_ref = zipfile.ZipFile(img_path+'.zip', mode='w', compression=zipfile.ZIP_DEFLATED)
j = 0
intensities = []
for i, subfolder in enumerate(folders):
    z = i*P['step']
    print('z = ', z, 'um')
    image_path = os.path.join(subfolder, 'images')

    ### Find trajectory file
    possible_traj_files = [os.path.join(subfolder, f) for f in os.listdir(subfolder)
                           if f.endswith('.txt') or f.endswith('.csv') or f.endswith('.tsv')]
    traj_filename = possible_traj_files[0]
    data = magic_load_trajectories(traj_filename)

    n_frames = data['frame'].nunique()

    ### Open movie
    movie = MovieFolder(image_path, auto_invert=True)

    ### Get image size
    image = movie.current_frame()
    width, height = image.shape[1], image.shape[0]

    previous_position = 0
    for image in tqdm.tqdm(movie.frames(), total=n_frames):
        data_frame = data[data['frame'] == previous_position]
        snippets = extract_cells(image, data_frame, image_size, crop=True) # border cells are kept
        intensities.extend([np.mean(snippet) for snippet in snippets])

        i = 0
        for _, row in data_frame.iterrows():
            j += 1

            # Make the label file
            row = pd.DataFrame([{'filename' : 'im{:05d}.png'.format(j), 'mean_z' : z}])
            df = pd.concat([df, row], ignore_index=True)

            # Save image
            filename = 'im{:05d}.png'.format(j)
            if P['zip']:
                image_bytes = io.BytesIO()
                imageio.imwrite(image_bytes, snippets[i], format='png')
                image_bytes.seek(0)  # Move the cursor to the start of the BytesIO object
                zip_ref.writestr(filename, image_bytes.read())  # Add image as 'image1.png'
            else:
                imageio.imwrite(os.path.join(img_path, filename, snippets[i]))

            i += 1

        previous_position = movie.position
if P['zip']:
    zip_ref.close()

P['normalization'] = float(1./np.mean(intensities))

## Save labels
df.to_csv(label_path, index=False)

## Save parameters
with open(parameter_path, 'w') as f:
    yaml.dump(P, f)
