from typing import Tuple

import PIL.Image
import numpy as np
import pandas as pd
from tqdm import tqdm

from ubc_ocean.config import BaseConfig


def compute_image_mean_std(config: BaseConfig, df: pd.DataFrame) -> Tuple[np.array, np.array]:
    """Compute per-channel mean and std using either thumbnails or TMA. Takes ~5min to run."""
    counter = 0
    sum_pixels = 0
    sum_pixel_val = 0.0
    sum_pixel_sq = 0.0
    for index, row in tqdm(df.iterrows()):
        image_id = row['image_id']
        if row['has_thumb']:
            counter += 1
            im = PIL.Image.open(config.path_train_thumbnails / f"{image_id}_thumbnail.png")
        elif row['is_tma']:
            counter += 1
            im = PIL.Image.open(config.path_train_images / f"{image_id}.png")
        else:
            raise NotImplementedError()
        im_arr = np.asarray(im).astype(np.float64) / 255.0
        sum_pixels += im_arr.shape[0] * im_arr.shape[1]
        sum_pixel_val += np.sum(im_arr, axis=(0, 1))
        sum_pixel_sq += np.sum(im_arr**2, axis=(0, 1))
        # print(index, im_arr.shape[0], im_arr.shape[1], im_arr.shape[0] * im_arr.shape[1])
        # print(np.mean(im_arr, axis=(0, 1)))
        # print(np.std(im_arr, axis=(0, 1)))
        # print(np.sum(im_arr, axis=(0, 1)))
        # print(np.sum(im_arr, axis=(0, 1)) / im_arr.shape[0] / im_arr.shape[1])
        # print(np.sqrt(np.sum((im_arr - np.mean(im_arr, axis=(0, 1)))**2, axis=(0, 1)) / im_arr.shape[0] / im_arr.shape[1]))
        # print(np.sqrt(np.sum(im_arr**2, axis=(0, 1)) / im_arr.shape[0] / im_arr.shape[1] - np.mean(im_arr, axis=(0, 1))**2))
        # if index > 5:
        #     break
    # print("End")
    # print(f"Per channel pixel value sum : {sum_pixel_val[0]:.0f} {sum_pixel_val[1]:.0f} {sum_pixel_val[2]:.0f}")
    mean = sum_pixel_val / sum_pixels
    std = np.sqrt(sum_pixel_sq / sum_pixels - mean**2)
    assert mean.shape == (3,)
    assert std.shape == (3,)
    print(f"Per channel pixel value mean: {mean[0]:.8f} {mean[1]:.8f} {mean[2]:.8f}")
    print(f"Per channel pixel value std : {std[0]:.8f} {std[1]:.8f} {std[2]:.8f}")
    print(f"{counter} images processed.")
    return mean, std


def compute_image_mean_std_tma(config: BaseConfig, df: pd.DataFrame) -> Tuple[np.array, np.array]:
    """Compute per-channel mean and std using only TMA. Takes ~30s to run."""
    counter = 0
    sum_pixels = 0
    sum_pixel_val = 0.0
    sum_pixel_sq = 0.0
    for index, row in tqdm(df.iterrows()):
        image_id = row['image_id']
        if row['is_tma']:
            counter += 1
            im = PIL.Image.open(config.path_train_images / f"{image_id}.png")
        else:
            continue
        im_arr = np.asarray(im).astype(np.float64) / 255.0
        sum_pixels += im_arr.shape[0] * im_arr.shape[1]
        sum_pixel_val += np.sum(im_arr, axis=(0, 1))
        sum_pixel_sq += np.sum(im_arr**2, axis=(0, 1))
    mean = sum_pixel_val / sum_pixels
    std = np.sqrt(sum_pixel_sq / sum_pixels - mean**2)
    assert mean.shape == (3,)
    assert std.shape == (3,)
    print(f"Per channel TMA pixels mean: {mean[0]:.8f} {mean[1]:.8f} {mean[2]:.8f}")
    print(f"Per channel TMA pixels std : {std[0]:.8f} {std[1]:.8f} {std[2]:.8f}")
    print(f"{counter} images processed.")
    return mean, std


def compute_image_mean_std_thumb(config: BaseConfig, df: pd.DataFrame) -> Tuple[np.array, np.array]:
    """Compute per-channel mean and std using only thumbnails. Takes ~5min to run."""
    counter = 0
    sum_pixels = 0
    sum_pixel_val = 0.0
    sum_pixel_sq = 0.0
    for index, row in tqdm(df.iterrows()):
        image_id = row['image_id']
        if row['has_thumb']:
            counter += 1
            im = PIL.Image.open(config.path_train_thumbnails / f"{image_id}_thumbnail.png")
        else:
            continue
        im_arr = np.asarray(im).astype(np.float64) / 255.0
        sum_pixels += im_arr.shape[0] * im_arr.shape[1]
        sum_pixel_val += np.sum(im_arr, axis=(0, 1))
        sum_pixel_sq += np.sum(im_arr**2, axis=(0, 1))
    mean = sum_pixel_val / sum_pixels
    std = np.sqrt(sum_pixel_sq / sum_pixels - mean**2)
    assert mean.shape == (3,)
    assert std.shape == (3,)
    print(f"Per channel thumbnail pixels mean: {mean[0]:.8f} {mean[1]:.8f} {mean[2]:.8f}")
    print(f"Per channel thumbnail pixels std : {std[0]:.8f} {std[1]:.8f} {std[2]:.8f}")
    print(f"{counter} images processed.")
    return mean, std
