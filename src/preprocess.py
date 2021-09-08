import re
from typing import List, Tuple, Union

import pandas as pd
from sklearn.utils import shuffle
from sklearn.compose import make_column_selector, make_column_transformer


# https://stackoverflow.com/questions/28986489/how-to-replace-text-in-a-column-of-a-pandas-dataframe/63139937
def clean_column_name(
    df: pd.DataFrame,
    to_remove: Union[str, List[str]],
    to_replace: Union[str, List[str]],
    value: str = "_",
) -> pd.DataFrame:
    """Replace unwanted characters in the dataframe column name

    Args:
        df (pd.DataFrame): the input dataframe
        to_remove (Union[str, List): a list or a single character that wants to be removed
        to_replace (Union[str, List): a list or a single character that wants to be replaced
        value (str, optional): what character to put to those being replaced. Defaults to "_".

    Returns:
        pd.DataFrame: the dataframe with column name corrected
    """
    if isinstance(to_remove, list):
        to_remove = "[" + re.escape("".join(to_remove)) + "]"
    df.columns = df.columns.str.replace(to_remove, "", regex=True)

    if isinstance(to_replace, list):
        to_replace = "[" + re.escape("".join(to_replace)) + "]"
    df.columns = df.columns.str.replace(to_replace, value, regex=True)

    return df


def downsample(
    df: pd.DataFrame,
    label_col: str,
    majority_class: int = 0,
    minority_class: int = 1,
    ratio: int = 2,
    seed: int = 36,
) -> pd.DataFrame:
    """Downsample the majority class to a certain ratio of
    the minority class for a binary classification

    Args:
        df (pd.DataFrame): the original dataframe
        label_col (str): the column that has the imbalanced class
        majority_class (int, optional): the value of majority class. Defaults to 0.
        minority_class (int, optional): the value of minority class. Defaults to 1.
        ratio (int, optional): the ratio between majority to minority class. Defaults to 2.
        seed (int, optional): the random state for shuffling. Defaults to 36.
    Returns:
        pd.Dataframe: the downsampled dataframe
    """
    df_minority = df.loc[df[label_col] == minority_class, :].reset_index(drop=True)
    df_majority = (
        df.loc[df[label_col] == majority_class, :]
        .sample(df_minority.shape[0] * ratio, random_state=seed)
        .reset_index(drop=True)
    )
    df_sampled = pd.concat([df_majority, df_minority], axis=0)
    return shuffle(df_sampled, random_state=36).reset_index(drop=True)


def get_preprocessor(cat_pipeline, num_pipeline):
    """A common pattern for preprocessing data with sklearn function"""
    preprocessor = make_column_transformer(
        (cat_pipeline, make_column_selector(dtype_include="object")),
        (num_pipeline, make_column_selector(dtype_include="number")),
    )
    return preprocessor


def do_transform(X: pd.DataFrame, preprocessor, col_name: list) -> pd.DataFrame:
    """Transform a Dataframe with fitted sklearn transformer and add column name back to it"""
    return pd.DataFrame(preprocessor.transform(X), columns=col_name)


def get_x_y(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe to features and labels"""
    X = df.drop([label_col], axis=1)
    y = df[label_col]
    return X, y
