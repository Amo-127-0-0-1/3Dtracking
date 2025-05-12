import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from concurrent.futures import ProcessPoolExecutor
import time
import tqdm

import imageio.v2 as iio
from PIL import Image, ImageFont, ImageDraw
from skimage import color

from gui.gui import ParametersDialog
from tracking.load_tracking import magic_load_trajectories

def tic(label):
    """
    Start a timer for the given label.
    """
    tic.timers[label] = time.time()

def toc(label):
    """
    Stop the timer for the given label and print elapsed time.
    """
    elapsed_time = time.time() - tic.timers[label]
    print(f"{label} took {elapsed_time:.2f} seconds.")


def overlay_frame(args):
    """
    Worker function to load one frame, overlay text for each cell, and return the result.
    Args tuple: (image_path, cells, invert, text_shift, font_path, font_size)
    """
    image_path, cells, text_shift, font_path, font_size = args

    # Load
    raw = iio.imread(image_path)
    if raw.ndim == 3:
        raw = raw[..., 0]  # Convert to grayscale if it's a color image
    # Convert to RGB for PIL
    rgb = color.gray2rgb(raw)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(font_path, font_size)

    # draw each z‐value
    for x, y, z in cells:
        xt = x - text_shift
        if xt < 0:
            xt = x + text_shift
        draw.text((xt, y), str(z), fill="red", font=font)  # Draw z-value

    return np.array(pil)


if __name__ == '__main__':
    # --- File dialogs ---
    root = tk.Tk()
    root.withdraw()

    frames_dir = filedialog.askdirectory(
        initialdir=os.path.expanduser('~/Downloads/'),
        title='Choose folder with unzipped frames'
    )
    traj_filename = filedialog.askopenfilename(
        initialdir=frames_dir,
        title='Choose a trajectory file'
    )
    name, _ = os.path.splitext(traj_filename)
    movie_out = name + '.mp4'

    # --- Parameters ---
    parameters = [
        ('pixel_size', 'Pixel size (µm)', 1.78),
        ('in_pixel', 'Trajectory in pixel?', True),
        ('fps', 'FPS (Hz)', 20.0),
        ('text_shift', 'Text shift (µm)', 200.0),
        ('font_size', 'Font size (pixel)', 30),
        ('truncate', 'Truncation (s)', 1e6),
    ]
    P = ParametersDialog(title='Enter parameters', parameters=parameters).value

    # conversion factors
    if P['in_pixel']:
        pixel_size = 1.0
        z_factor = P['pixel_size'] * 1.33
    else:
        pixel_size = P['pixel_size']
        z_factor = 1.33
    fps = P['fps']
    text_shift = int(P['text_shift'] / P['pixel_size'])
    font_size = int(P['font_size'])

    tic.timers = {}
    tic('TOTAL')

    # --- Load trajectories ---
    tic('Load and process trajectories')
    data = magic_load_trajectories(traj_filename)
    data = data[data['frame'] / fps < P['truncate']]
    toc('Load and process trajectories')

    # --- sorted list of image paths ---
    tic('Build image paths')
    all_files = [
        f for f in os.listdir(frames_dir)
        if f.lower().endswith(('.tif', '.tiff'))
    ]
    def _num_key(fn):
        name, _ = os.path.splitext(fn)
        return int(name) if name.isdigit() else name
    image_files = sorted(all_files, key=_num_key)
    image_paths = [os.path.join(frames_dir, f) for f in image_files]
    # Group trajectories by frame
    frame_groups = {frm: df for frm, df in data.groupby('frame')}
    toc('Build image paths')

    # --- Prepare tasks for parallel overlay ---
    tic('Prepare tasks')
    tasks = []
    for n, path in enumerate(image_paths):
        X = []
        df = frame_groups.get(n)
        if df is not None:
            for row in df.itertuples():
                x_px = int(row.x // pixel_size)
                y_px = int(row.y // pixel_size)
                z_px = int(row.z * z_factor)
                X.append((x_px, y_px, z_px))
        tasks.append((path, X, text_shift, "arial.ttf", font_size))
    toc('Prepare tasks')

    # --- Write movie parallel workers ---
    tic('Write movie parallel')
    writer = iio.get_writer(movie_out, fps=fps, quality=5)

    with ProcessPoolExecutor() as executor:
        for img in tqdm.tqdm(executor.map(overlay_frame, tasks), total=len(tasks), desc="Processing frames"):
            writer.append_data(img)

    writer.close()
    toc('Write movie parallel')

    print(f"Done! Saved overlay movie to {movie_out}")
    toc('TOTAL')

# 11797
# Load and process trajectories took 0.15 seconds.
# Build image paths took 0.31 seconds.
# Prepare tasks took 6.53 seconds.
# Write movie parallel took 195.63 seconds.
# TOTAL took 202.63 seconds.

