import numpy as np
import pandas as pd

from src.eda import see_missing, get_freq_table


def test_see_missing():
    df_in = pd.DataFrame(
        {
            "col1": [1, 2, 3, np.nan],
            "col2": [1, 3, 4, 7.1],
            "col3": [np.nan, np.nan, "ha", "hey"],
        }
    )
    df_test = see_missing(df_in)
    df_true = pd.DataFrame(
        {
            "columns": ["col3", "col1", "col2"],
            "total_missing": [2, 1, 0],
            "percent_missing": [50.0, 25.0, 0.0],
        }
    )
    pd.testing.assert_frame_equal(df_true, df_test)


def test_get_freq_table():
    df_in = pd.DataFrame({"col1": ["a"] * 2 + ["b"] * 3, "col2": [1] * 5})
    df_test = get_freq_table(df_in, col="col1")
    df_true = pd.DataFrame({"count": [3, 2], "percentage": [0.6, 0.4]})
    df_true.index = ["b", "a"]
    pd.testing.assert_frame_equal(df_true, df_test)
