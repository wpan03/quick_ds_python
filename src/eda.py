from numpy.lib import stride_tricks
import pandas as pd


def see_missing(df: pd.DataFrame, only_missing: bool = False) -> pd.DataFrame:
    """Show the number and percentage of missing values in each column"""
    total_missing = df.isnull().sum().values
    percent_missing = total_missing * 100 / len(df)

    df_missing_info = pd.DataFrame({'columns': df.columns,
                                    'total_missing': total_missing,
                                    'percent_missing': percent_missing})
    df_missing_info = df_missing_info.sort_values(
        'percent_missing', ascending=False, ignore_index=True)

    if only_missing:
        return df_missing_info.query('total_missing > 0')
    return df_missing_info

def get_freq_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Get the count and percentage of each unique value in the column"""
    num_count = df[col].value_counts()
    perc_count = df[col].value_counts(normalize=True)
    df_sum = pd.concat([num_count, perc_count], axis=1)
    df_sum.columns = ['count', 'percentage']
    return df_sum


def plot_corr_heatmap(df: pd.DataFrame) -> None:
    """Plot the correlation matrix of a dataframe in heatmap"""
    corr_matrix = df.corr()
    corr_matrix.style.background_gradient(cmap='coolwarm').set_precision(2)
