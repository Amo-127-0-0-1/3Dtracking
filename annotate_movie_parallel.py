import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from concurrent.futures import ProcessPoolExecutor
import time

import imageio.v2 as iio
from PIL import Image, ImageFont, ImageDraw
from skimage import color

from gui.gui import ParametersDialog
from tracking.load_tracking import magic_load_trajectories
from tqdm import tqdm


def tic(label):
    tic.timers[label] = time.time()

def toc(label):
    elapsed_time = time.time() - tic.timers[label]
    print(f"{label} took {elapsed_time:.2f} seconds.")

def overlay_frame(args):
    """
    Worker function to load one frame, overlay text for each cell, and return the result.
    """
    image_path, cells, text_shift, font_path, font_size = args

    raw = iio.imread(image_path)
    if raw.ndim == 3:
        raw = raw[..., 0]  # Convert to grayscale if it's a color image

    rgb = color.gray2rgb(raw)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(font_path, font_size)

    for x, y, z in cells:
        xt = x - text_shift
        if xt < 0:
            xt = x + text_shift
        draw.text((xt, y), str(z), fill="red", font=font)

    return np.array(pil)

def make_task(path, frame_idx, frame_groups, pixel_size, z_factor, text_shift, font_path, font_size):

    coords = []
    df = frame_groups.get(frame_idx)
    if df is not None:
        for row in df.itertuples():
            x_px = int(row.x // pixel_size)
            y_px = int(row.y // pixel_size)
            z_px = int(row.z * z_factor)
            coords.append((x_px, y_px, z_px))
    return (path, coords, text_shift, font_path, font_size)

def main():
    # --- dialogs ---
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
    name, _ = os.path.splitext(os.path.basename(traj_filename))
    movie_out = os.path.join(os.path.dirname(traj_filename), name + '.mp4')


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

    # Conversion factors
    if P['in_pixel']:
        pixel_size = 1.0
        z_factor = P['pixel_size'] * 1.33
    else:
        pixel_size = P['pixel_size']
        z_factor = 1.33
    fps = P['fps']
    text_shift = int(P['text_shift'] / P['pixel_size'])
    font_size = int(P['font_size'])

    # --- Timing ---
    tic.timers = {}
    tic('TOTAL')

    # --- Load trajectories ---
    tic('Load and process trajectories')
    data = magic_load_trajectories(traj_filename)
    data = data[data['frame'] / fps < P['truncate']]
    frame_groups = {frm: df for frm, df in data.groupby('frame')}
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
    toc('Build image paths')

    # --- Chunked parallel processing ---
    chunk_size = 10000           # Number of frames per chunk
    max_workers = None         # Defaults to os.cpu_count()

    tic('Write movie chunked')
    writer = iio.get_writer(movie_out, fps=fps, quality=5)
    total_frames = len(image_paths)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for start in range(0, total_frames, chunk_size):
            end = min(start + chunk_size, total_frames)
            sub_paths = image_paths[start:end]

            # tasks for this chunk, preserving global frame indices
            tasks = [
                make_task(path, idx, frame_groups,
                          pixel_size, z_factor,
                          text_shift, "arial.ttf", font_size)
                for idx, path in enumerate(sub_paths, start)
            ]

            for img in tqdm(executor.map(overlay_frame, tasks), total=len(tasks),
                            desc=f"Processing frames {start}-{end - 1}"):
                writer.append_data(img)

    writer.close()
    toc('Write movie chunked')

    print(f"Done! Saved overlay movie to {movie_out}")
    toc('TOTAL')

if __name__ == '__main__':
    main()
