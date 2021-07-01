import pandas as pd
from sklearn.utils import shuffle

def downsample(df: pd.DataFrame,
               label_col: str,
               majority_class: int = 0,
               minority_class: int = 1,
               ratio: int = 2,
               seed: int = 36) -> pd.DataFrame:
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
    df_minority = df.query('label_col = @minority_class').reset_index(drop=True)
    df_majority = df.query('label_col = @majority_class')\
                    .sample(df_minority.shape[0] * ratio)\
                    .reset_index(drop=True)
    df_sampled = pd.concat([df_majority, df_minority], axis=0)
    return shuffle(df_sampled, random_state=36).reset_index(drop=True)