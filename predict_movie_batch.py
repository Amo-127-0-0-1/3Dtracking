"""
only with unzipped frames
"""
import os
import time
from collections import defaultdict

import numpy as np
import tqdm
import tkinter as tk
from tkinter import filedialog

import imageio.v2 as imageio
from tensorflow.keras.models import load_model

from gui.gui import ParametersDialog
from tracking.load_tracking import magic_load_trajectories
from movie.cell_extraction import extract_cells

# --- Constants ---
MAX_FRAMES_IN_MEMORY = 10000
BATCH_SIZE = 1000

# --- Timing helpers ---
timers = {}
def tic(label):
    timers[label] = time.time()
def toc(label):
    timers[label] = time.time() - timers[label]

# --- frame loader ---
def img_loader(image_paths, gray=False, auto_invert=False):
    # Decide invert on the first frame
    first = imageio.imread(image_paths[0])
    if gray and first.ndim == 3:
        first = first[:, :, 0]
    invert = False
    if auto_invert:
        # White background if mean > half of max dtype
        if first.mean() > np.iinfo(first.dtype).max / 2:
            invert = True

    def load_frame(path):
        img = imageio.imread(path)
        if gray and img.ndim == 3:
            img = img[:, :, 0]
        if invert:
            img = np.iinfo(img.dtype).max - img
        return img

    return load_frame

# --- sort key for filenames ---
def _num_key(fn):
    name, _ = os.path.splitext(fn)
    try:
        return int(name)
    except ValueError:
        return name

# --- file dialogs ---
root = tk.Tk()
root.withdraw()

movie_filename = filedialog.askopenfilename(
    initialdir=os.path.expanduser('~/Downloads/'),
    title='Choose a movie file'
)
traj_filename = filedialog.askopenfilename(
    initialdir=os.path.dirname(movie_filename),
    title='Choose a trajectory file'
)
model_dir = filedialog.askdirectory(
    initialdir=os.path.dirname(movie_filename),
    title='Choose a model folder'
)
output_csv = os.path.splitext(traj_filename)[0] + '_with_z.csv'

# --- Parameter dialog ---
parameters = [
    ('in_pixel',   'Trajectory already in pixel units?', True),
    ('pixel_size', 'Pixel size (µm)',               1.78),
    ('normalize',  'Normalize each snippet?',        False),
]
P = ParametersDialog(
    parent=root,
    title='Enter parameters',
    parameters=parameters
).value

pixel_size = 1.0 if P['in_pixel'] else P['pixel_size']

# --- Start total timer ---
tic('total')

# --- Load Keras model ---
tic('model_loading')
model = load_model(
    model_dir,
    custom_objects={
        'modified_mae': None,
        'mean_abs_difference_metric': None,
        'combined_loss': None
    }
)
# input_shape (batch, H, W, C)
_, IMG_H, IMG_W, IMG_C = model.input_shape
toc('model_loading')

# --- Load trajectories ---
tic('trajectory_loading')
data = magic_load_trajectories(traj_filename)
data['z'] = np.nan
toc('trajectory_loading')

# --- sorted list of TIFFs ---
tic('movie_loading')
if movie_filename.lower().endswith('.zip'):
    raise ValueError("ZIP inputs not supported—please select a folder of TIFFs.")
img_dir = os.path.dirname(movie_filename)
all_files = [
    f for f in os.listdir(img_dir)
    if f.lower().endswith(('.tif', '.tiff'))
]
image_files = sorted(all_files, key=_num_key)
image_paths = [os.path.join(img_dir, f) for f in image_files]
n_frames = len(image_paths)

# as in MovieFolder(gray=True, auto_invert=True)
load_frame = img_loader(
    image_paths,
    gray=True,
    auto_invert=True
)
toc('movie_loading')
print(f"Total frames: {n_frames}")

# --- Chunked, batched prediction loop ---
chunk_stats = []
for chunk_start in tqdm.trange(0, n_frames, MAX_FRAMES_IN_MEMORY, desc="Chunks"):
    label = f'chunk_{chunk_start}'
    tic(label + '_total')
    tic(label + '_load')

    snippets = []
    frame_idxs = []
    chunk_end = min(chunk_start + MAX_FRAMES_IN_MEMORY, n_frames)

    # load & extract cells
    for fi in range(chunk_start, chunk_end):
        subset = data[data['frame'] == fi]
        if subset.empty:
            continue
        img = load_frame(image_paths[fi])
        cells = extract_cells(img, subset, IMG_H, pixel_size=pixel_size)

        if P['normalize']:
            cells = [
                c.astype(np.float32) / (np.mean(c) + 1e-8)
                for c in cells
            ]
        else:
            cells = [c.astype(np.float32) for c in cells]

        snippets.extend(cells)
        frame_idxs.extend([fi] * len(cells))

    toc(label + '_load')

    if not snippets:
        # no frames in this chunk
        timers[f'{label}_predict'] = 0.0
        timers[f'{label}_assign']  = 0.0
        toc(label + '_total')
        chunk_stats.append((chunk_start, 0,0,0, timers[f'{label}_total']))
        continue

    # shape
    X = np.stack(snippets, axis=0)
    if X.ndim == 3:
        X = X[..., np.newaxis]  # add channel dim

    # batched predict
    tic(label + '_predict')
    preds_by_frame = defaultdict(list)
    for i in range(0, len(X), BATCH_SIZE):
        batch = X[i:i+BATCH_SIZE]
        batch_preds = model.predict(batch, verbose=0)
        for frm, p in zip(frame_idxs[i:i+len(batch_preds)], batch_preds):
            zval = float(p) if np.ndim(p)==0 else float(p[0])
            preds_by_frame[frm].append(zval)
    toc(label + '_predict')

    # assign back
    tic(label + '_assign')
    for frm, zlist in preds_by_frame.items():
        rows = data.index[data['frame'] == frm].tolist()
        if len(rows) != len(zlist):
            raise RuntimeError(
                f"Frame {frm}: {len(rows)} rows but {len(zlist)} preds"
            )
        for idx, zval in zip(rows, zlist):
            data.at[idx, 'z'] = zval
    toc(label + '_assign')

    toc(label + '_total')
    chunk_stats.append((
        chunk_start,
        timers[f'{label}_load'],
        timers[f'{label}_predict'],
        timers[f'{label}_assign'],
        timers[f'{label}_total']
    ))

# --- Save ---
tic('save_csv')
data.to_csv(output_csv, index=False)
toc('save_csv')
toc('total')

print("\n=== Timing Summary ===")
print(f"Model load:      {timers['model_loading']:.2f}s")
print(f"Trajectory load: {timers['trajectory_loading']:.2f}s")
print(f"Movie load:      {timers['movie_loading']:.2f}s")
print(f"Pred loop:       {sum(s[-1] for s in chunk_stats):.2f}s")
print(f"CSV save:        {timers['save_csv']:.2f}s")
print(f"Total run:       {timers['total']:.2f}s\n")

print(f"{'Chunk':>6} | {'Load':>5} | {'Pred':>5} | {'Assn':>5} | {'Total':>5}")
for st, ld, pr, asn, tot in chunk_stats:
    print(f"{st:6d} | {ld:5.2f} | {pr:5.2f} | {asn:5.2f} | {tot:5.2f}")



# Total frames: 11798
# === Timing Summary ===
# Model load:      1.16s
# Trajectory load: 0.13s
# Movie load:      0.06s
# Pred loop:       97.34s
# CSV save:        0.88s
# Total run:       99.58s
#
#  Chunk |  Load |  Pred |  Assn | Total
#      0 | 61.80 | 13.37 |  1.73 | 77.57
#  10000 | 17.03 |  2.30 |  0.32 | 19.77

