from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


parameter_grids = {'bootstrap': [True, False],
                   'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                   'max_features': ['auto', 'sqrt', 0.5],
                   'min_samples_leaf': [1, 2, 4],
                   'min_samples_split': [2, 5, 10],
                   'n_estimators': [50, 100, 200, 600, 800, 1000]}


def rf(X, y, n_estimators=40, max_samples=200000,
       max_features='auto', min_samples_leaf=5, **kwargs):

    return RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators,
                                  max_samples=max_samples, max_features=max_features,
                                  min_samples_leaf=min_samples_leaf).fit(X, y)


def grid_search(mod_rf, parameter_grids):
    rf_grid = GridSearchCV(estimator=mod_rf, param_distributions=parameter_grids,
                           n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
    return rf_grid
