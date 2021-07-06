import pandas as pd


def get_lg_coef(df: pd.DataFrame, mod_lg) -> pd.DataFrame:
    """Get and sort the coefficent from logistic regression"""
    df_coef = pd.DataFrame({'name': df.columns,
                            'value': mod_lg.coef_.reshape(-1)})
    return df_coef.sort_values('value', ignore_index=True)


def get_feature_imp(df: pd.DataFrame, mod_rf) -> pd.DataFrame:
    """Get the feature importance of random forest model"""
    result = pd.DataFrame({'name': df.columns,
                           'score': mod_rf.feature_importances_})

    return result.sort_values('score', ascending=False, ignore_index=True)
