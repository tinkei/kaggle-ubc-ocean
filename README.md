# UBC-OCEAN

My code used in the Kaggle competition "[UBC Ovarian Cancer Subtype Classification and Outlier Detection (UBC-OCEAN)](https://www.kaggle.com/competitions/UBC-OCEAN)".

## Usage

The source code is mounted as a "Kaggle Dataset", added to path via `sys.path.append("/kaggle/input/ubc-ocean-code/src/")` in your Kaggle Notebook.

Dependencies that are not part of Kaggle's image has to be mounted as other datasets and installed prior to submission,
as the submitted notebook will not have internet access.

## Approaches

- Image hash
  - Hamming distance via image hash
  - Image hash as features to LGBM
    - Implementation incomplete
- Pretrained models from TIMM
  - Efficient Net V2: "efficientnetv2_rw_s.ra2_in1k"
    - Final submission
    - Ranked #746 out of 1327 teams
    - Balanced accuracy 32%
  - Vision Transformer: "vit_small_patch16_384.augreg_in21k_ft_in1k"
    - Poor performance (my bad)
- Segmentation models from Torch
  - DeepLabV3 with MobileNet V3 backbone
    - 3:7 train:test split (not a typo)
    - Balanced validation score at 61.19% (~#8 in public leaderboard, ~#2 in private, in theory)
    - Best performing method (in my head)
    - **_Submission failure!!!_** 😭

## Dataloading

- Thumbnails
  - Use thumbnails resized to a fixed resolution.
- Tiles
  - Decompose full resolution images into tiles, similar to what's demonstrated here: https://www.kaggle.com/code/jirkaborovec/cancer-subtype-decompose-wsi-tiles-0-25x
- Random sampling of tiles
  - For 5 different magnifications, sample 20 random tiles each from the full resolution images, then downsize by 2x or 4x.

When operating on thumbnails, the most expensive step in the transformation pipeline is the image resize step.
Therefore, in training, the resized images are precomputed and mounted as a "Kaggle Dataset".
Only in the test step, when there is a "cache miss", will images be resized on-the-fly.

When operating on full resolution images, the most time and memory intensive step is opening the image file itself.
For this, we utilized the `pyvips` library to build a compute graph, to load only specific areas of the image file.

While we already sampled tiles from the source images,
the data loader will again sample random patches from a resized thumbnail/tile.
In this way, the neural network can have different views of the same image, with fairly low compute.
This is because most pretrained models are trained on images of 244px,
which is a tiny resolution compared with the thumbnails (4000x4000px) or the source images (50000x50000+px).
The average or median prediction of all sampled patches will be returned by the model.
The median prediction is used, as part of the image may contain healthy cells, or irrelevant features.

We use certain quality factors to select random image patches:
- Image patch/tile is not background (R+G+B = 0)
- Image patch/tile is not interstitial space (R+G+B <= 240*3)
- Image patch/tile does not overlap with previously selected samples

We ditch our initial dataset mean/std values, and instead aggregate the statistics _after_ the quality factors are applied.
The final mean/std are very similar for TMA images vs others.

## Nice to haves:

- Ensembling
- Finish the image hash baseline
- Handle different magnification of TMA images
