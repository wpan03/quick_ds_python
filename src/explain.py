import pandas as pd

def get_feature_imp(df: pd.DataFrame, mod_rf) -> pd.DataFrame:
    """Get the feature importance of random forest model"""
    result = pd.DataFrame({'name':df.columns,
                          'score':mod_rf.feature_importances_})
    
    return result.sort_values('score', ascending=False, ignore_index=True)