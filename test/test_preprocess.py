import pandas as pd

from src.preprocess import clean_column_name


def test_clean_column_name():
    df_in = pd.DataFrame(
        {
            "(Column 1": [1, 2],
            "Column 2": [3, 3],
            "Column\n3": [5, 5],
            "Column_4": [6, 6],
        }
    )
    df_test = clean_column_name(df_in, to_remove="(", to_replace=[" ", "\n"])
    df_true = pd.DataFrame(
        {
            "Column_1": [1, 2],
            "Column_2": [3, 3],
            "Column_3": [5, 5],
            "Column_4": [6, 6],
        }
    )

    pd.testing.assert_frame_equal(df_true, df_test)
