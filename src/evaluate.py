from typing import Union
import numpy as np
import pandas as pd
from sklearn import metrics

def binary_evaluation(X: Union[pd.DataFrame, np.ndarray], 
                      y: Union[pd.Series, np.ndarray], 
                      mod) -> pd.DataFrame:
    """Evaluate the performance of a binary classification model"""
    
    pred = mod.predict(X)
    
    acc = metrics.accuracy_score(y, pred)
    prec = metrics.precision_score(y,pred)
    rec = metrics.recall_score(y,pred)
    f1 = metrics.f1_score(y, pred)
    
    df_result = pd.DataFrame(columns=['metrics', 'value'])
    df_result['metrics'] = ['accuracy', 'precision', 'recall', 'f1']
    df_result['value'] = [acc, prec, rec, f1]
    
    return df_result