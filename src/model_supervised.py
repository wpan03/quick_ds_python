from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV



def rf(X, y, n_estimators=40, max_samples=200000,
       max_features='auto', min_samples_leaf=5, **kwargs):

    return RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators,
                                  max_samples=max_samples, max_features=max_features,
                                  min_samples_leaf=min_samples_leaf).fit(X, y)


def grid_search(mod_rf, parameter_grids: dict):
    rf_grid = GridSearchCV(estimator=mod_rf, param_distributions=parameter_grids,
                           n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
    return rf_grid
