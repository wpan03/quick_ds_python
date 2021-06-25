from typing import Union

import numpy as np
import pandas as pd
import sklearn 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline



def rf(X, y, n_estimators=40, max_samples=200000,
       max_features='auto', min_samples_leaf=5, **kwargs):

    return RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators,
                                  max_samples=max_samples, max_features=max_features,
                                  min_samples_leaf=min_samples_leaf).fit(X, y)


def grid_search(X: Union[pd.DataFrame, np.ndarray],
                y: Union[pd.Series, np.ndarray],
                mod,
                grids: dict, 
                preprocessor=None,
                num_fold: int = 3,
                **kwargs) -> sklearn.model_selection._search.GridSearchCV:
    if preprocessor is not None:
        clf = make_pipeline(preprocessor, mod)
    else:
        clf = mod
    gs = GridSearchCV(clf, cv=num_fold, param_grid=grids, **kwargs)
    _ = gs.fit(X, y)
    return gs
