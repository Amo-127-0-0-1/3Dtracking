# Notes
Batch prediction (predict_movie_batch.py) suitable for offline processing with powerful GPU, and parallel video annotation (annotate_movie_parallel.py).

# Three-dimensional tracking of protists from two-dimensional microscopy images

Code for the article:

Three-dimensional tracking of protists from two-dimensional microscopy images (2025)

Ali Hosseini, Célia Fosse, Maya Awada, Marcel Stimberg, Romain Brette

## Requirements
- `tensorflow`
- `keras`
- `imageio` (for reading images and movies)
- `tqdm` (progress bar)
- `PIL` (for movie annotation)
- `imageio[ffmpeg]` (writing movies)

## 2D tracking
This code assumes that 2D tracks have already been extracted from the movie. This can be done with various tools,
such as Fasttrack or Fiji.
The trajectory file can be a Fasttrack file, or a simple csv/tsv file with fields x, y and frame.
Dimensions can be in pixel or in um.

## Typical workflow
1. Make a training data set. This can be done with:
  - `dataset/create_dataset.py` for a movie on a tilted slide;
  - `dataset/create_stack_dataset.py` for movies on horizontal slides at several known z positions;

2. Train with `train.py`. Training can be with respect to z or, in the case of a synthetic dataset, sigma
(the amount of blurring).

3. Make predictions on a movie (`predict_movie.py`) or on a dataset (`predict_dataset.py`).

4. Evaluate predictions on a dataset with `evaluation/evaluate_tilted_slide.py` (tilted dataset),
`evaluation/evaluate_stack` (stack dataset) or `evaluate_synthetic` (synthetic dataset).

## File structure

- `train.py` trains the model on a labeled dataset.
- `predict_dataset.py` applies the trained model on a dataset and adds a column `z_predict`.
- `predict_movie.py` applies the trained model on a tiff folder (movie) with trajectory file,
   and adds a column `z` to the trajectory file.
- `annotate_movie.py` writes the estimated z next to each cell. Must be used with a trajectory file with estimate 
   z, produced by `predict_movie.py`.

### `augmentation/`
Various image augmentation functions for training.

### `dataset/`
Scripts to create labeled datasets of cell images from a tiff folder (movie)
and a trajectory file.

### `evaluation/`
Scripts to evaluate the quality of estimation.

### `figures/`
Scripts to make the figures of the paper.

### `gui/`
These are utility scripts for user interfaces.

### `models/`
Various estimation models.

### `synthetic/`
Various scripts to create and analyze synthetic data.

### `tracking/`
These are utility scripts to load and analyze trajectory files.

### `trained/`
Trained models. Note that the model must be trained on each specific experimental setup.
