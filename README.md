# UBC-OCEAN

My code for "UBC Ovarian Cancer Subtype Classification and Outlier Detection (UBC-OCEAN)" Kaggle competition: https://www.kaggle.com/competitions/UBC-OCEAN

## Approaches

- Image hash
  - Hamming distance via image hash
  - Image hash as features to LGBM
    - Implementation incomplete
- Pretrained CNN
  - Efficient Net V2: "efficientnetv2_rw_s.ra2_in1k"
    - Final submission
    - Ranked 746/1327
    - Balanced accuracy 32%
  - Vision Transformer: "vit_small_patch16_384.augreg_in21k_ft_in1k"
    - Poor performance (my bad)
- Segmentation
  - Best performing method
  - 3:7 train:test split (not a typo) with validation score at 60%, but submission failure

## Dataloading

- Thumbnails
- Tiles
- Random sampling of tiles

## Nice to haves:

- Ensembling
- Finish the image hash baseline
- Handle different magnification of TMA images
