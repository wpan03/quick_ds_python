from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline


def binary_evaluation(X: Union[pd.DataFrame, np.ndarray],
                      y: Union[pd.Series, np.ndarray],
                      mod) -> pd.DataFrame:
    """Evaluate the performance of a binary classification model"""

    pred = mod.predict(X)

    acc = metrics.accuracy_score(y, pred)
    prec = metrics.precision_score(y, pred)
    rec = metrics.recall_score(y, pred)
    f1 = metrics.f1_score(y, pred)

    df_result = pd.DataFrame(columns=['metrics', 'value'])
    df_result['metrics'] = ['accuracy', 'precision', 'recall', 'f1']
    df_result['value'] = [acc, prec, rec, f1]

    return df_result


def binary_cross_evaluate(X: Union[pd.DataFrame, np.ndarray],
                          y: Union[pd.Series, np.ndarray],
                          mod,
                          preprocessor=None,
                          num_fold: int = 3,
                          **kwargs) -> pd.DataFrame:
    """Cross validation with accuracy and f1 for binary classification

    Args:
        X (Union[pd.DataFrame, np.ndarray]): input features
        y (Union[pd.Series, np.ndarray]): label
        mod (sklearn classifier): a not fitted sklearn classifier model
        preprocessor (sklearn preprocess, optional): preprocessing for X. Defaults to None.
        num_fold (int, optional): number of folds for cross validation. Defaults to 3.
        **kwargs (optional): additional arguments for the cross_validate method

    Returns:
        pd.DataFrame: the result of cross validation
    """
    if preprocessor is not None:
        clf = make_pipeline(preprocessor, mod)
    else:
        clf = mod
    result_dict = cross_validate(clf, X, y, cv=num_fold, scoring=['accuracy', 'f1'], **kwargs)
    return pd.DataFrame(result_dict)

def plot_cf(mod,
            X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray],
            title: int = 'Confusion Matrix') -> None:
    disp = metrics.plot_confusion_matrix(mod, X, y, cmap=plt.cm.Blues)
    disp.ax_.set_title(title)
    plt.show()