from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


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


def get_permute_imp_df(mod, X: pd.DataFrame, y: Union[pd.Series, np.ndarray],
                       return_full_result: bool = False,
                       **kwargs) -> Union[pd.DataFrame, tuple]:
    """Return the result from permutation importance

    Args:
        mod (sklearn model estimator): An estimator that has already been fitted and is compatible with scorer
        X (pd.DataFrame): Data on which permutation importance will be computed.
        y (Union[pd.Series, np.ndarray, None]): Targets for supervised or None for unsupervised.
        return_full_result (bool, optional): if true, return both the formatted dataframe and the full result. 
                                             Defaults to False.

    Returns:
        Union[pd.DataFrame, tuple]: a formatted permutation result df or with a the full result
    """
    result = permutation_importance(mod, X, y, **kwargs)
    df_result = pd.DataFrame({'name': X.columns,
                              'importance_mean': result.importances_mean,
                              'importance_std': result.importances_std})
    df_result = df_result.sort_values(
        'importance_mean', ascending=False, ignore_index=True)
    if return_full_result:
        return df_result, result
    return df_result


# Ref: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html
def plot_permute_imp(X: pd.DataFrame, result) -> None:
    """Make a box plot from the permutation importance result"""
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=X.columns[sorted_idx])
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    plt.show()


def cross_entropy_loss(y_true: np.ndarray, y_pred_1: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Calculate the binary cross entropy loss for each observation, no reduction"""
    y_pred_1 = np.clip(y_pred_1, eps, 1 - eps)
    loss = -(y_true * np.log(y_pred_1) + (1 - y_true) * np.log(1 - y_pred_1))
    return loss


def get_binary_error_analysis(X: pd.DataFrame, y: pd.Series, mod, only_show_error: bool = False) -> pd.DataFrame:
    """Calculate the loss for each row withe actual label and predicted label"""
    df_result = X.copy(deep=True)
    pred_prob = mod.predict_proba(X)
    pred_class = mod.predict(X)
    loss = cross_entropy_loss(y.values, pred_prob[:, 1])
    df_result['class1_prob'] = pred_prob[:, 1]
    df_result['pred_class'] = pred_class
    df_result['actual_class'] = y.values
    df_result['loss'] = loss
    if only_show_error:
        df_result = df_result[df_result['pred_class']
                              != df_result['actual_class']]
    return df_result.sort_values('loss', ascending=False, ignore_index=True)
