rf_grids = {'max_features': ['auto', 'sqrt', 0.5],
            'min_samples_leaf': [1, 3, 5, 7, 9],
            'n_estimators': [50, 100, 200, 400]}

hgb_grids = {'learning_rate': [0.05, 0.1, 0.2],
             'max_iter': [50, 100, 200],
             'min_samples_leaf': [10, 20, 30],
             'l2_regularization': [0, 0.01, 0.1]
             }
