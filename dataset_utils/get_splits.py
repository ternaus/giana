"""
Split data into folds.

We have 18 videos with numbers 1..18

Functions in the file help to split data into folds.
"""
from pathlib import Path

from sklearn.model_selection import KFold
import numpy as np


def collect_image_paths(video_ids, data_path: Path):
    """Gets the list of the image paths in a given videos.

    Args:
        video_ids:
        data_path (Path): Path where images are stored.

    Returns:

    """
    file_paths = []

    for video_id in video_ids:
        image_path = data_path.joinpath(video_id).joinpath('images')
        file_paths += list(image_path.glob('*.png'))

    return file_paths


def get_train_val_image_paths(data_path: Path, fold_id: 0, num_splits: int = 5):
    """Returns list of image paths to files in train in val for a given number of folds and fold id.

    Args:
        data_path (Path): Path to the data.
        fold_id (int): Desired fold id.
        num_splits (int): Number of folds.

    Returns: (list of paths to train images, list of paths to val images)

    """
    assert 0 <= fold_id < num_splits

    video_ids = np.array([x.name for x in (data_path.glob('*'))])

    kf = KFold(n_splits=num_splits, random_state=42)

    train_video_ids, val_video_ids = list(kf.split(video_ids))[fold_id]

    train_file_names = collect_image_paths(video_ids[train_video_ids], data_path)

    val_file_names = collect_image_paths(video_ids[val_video_ids], data_path)

    return train_file_names, val_file_names


if __name__ == '__main__':
    get_train_val_image_paths(Path('/home/vladimir/workspace/data_fast/giana/train'), 0)
