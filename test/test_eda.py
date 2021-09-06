import numpy as np
import pandas as pd

from src.eda import see_missing


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
