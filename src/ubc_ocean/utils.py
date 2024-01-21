import pandas as pd


def get_total_model_params(model):
    return sum(p.numel() for p in model.parameters())


def get_trainable_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def preprocess_training_data(df: pd.DataFrame, use_other: bool = False) -> pd.DataFrame:
    """Add transformations according to pathologist advice, and sampling weights.

    Modification is in-place.
    Source: https://www.kaggle.com/competitions/UBC-OCEAN/discussion/445804
    """

    exclusion_list = [281, 3222, 5264, 9154, 12244, 26124, 31793, 32192, 33839, 41099, 52308, 54506, 63836]
    deletion_list = [1289, 32035]
    emphasize_list = [34822]

    df.loc[df["image_id"] == 15583, "label"] = "MC"
    if use_other:
        df.loc[df["image_id"].isin(exclusion_list), "label"] = "Other"
    else:
        df.drop(index=df.loc[df["image_id"].isin(exclusion_list)].index, inplace=True)
    df.drop(index=df.loc[df["image_id"].isin(deletion_list)].index, inplace=True)

    df["weight"] = 1 / df.groupby("label")["label"].transform("count")
    df["weight"] = df["weight"] / (1 / df["label"].value_counts()).sum()
    df.loc[df["image_id"].isin(emphasize_list), "weight"] *= 2
    # df.loc[df["has_mask"], "weight"] /= 2
    df.reset_index(drop=True, inplace=True)

    return df
