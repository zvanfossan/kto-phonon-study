import os, json
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.tree import DecisionTreeRegressor
seed = np.random.seed(10)

def get_rmse(actual, pred):
    return np.mean([(actual[i]-pred[i])**2 for i in range(len(actual))])**0.5

def run_cv(n_folds, model, X_train, y_train, stratify=False):
    """
    Args:
        n_folds (int) : how many folds of CV to do
        model (sklearn Model) : what model do we want to fit
        X_train (np.array) : feature matrix
        y_train (np.array) : target array
        stratify (bool) : if True, use stratified CV, otherwise, use random CV
        
    Returns:
        a dictionary with scores from each fold for training and validation
            {'train' : [list of training scores],
             'val' : [list of validation scores]}
            - the length of each list = n_folds
    """
    if stratify:
        folds = StratifiedKFold(n_splits=n_folds).split(X_train, y_train)
    else:
        folds = KFold(n_splits=n_folds).split(X_train, y_train)

    train_scores, val_scores = [], []
    for k, (train, val) in enumerate(folds):

        X_train_cv = X_train[train]
        y_train_cv = y_train[train]

        X_val_cv = X_train[val]
        y_val_cv = y_train[val]

        model.fit(X_train_cv, y_train_cv)

        y_train_cv_pred = model.predict(X_train_cv)
        y_val_cv_pred = model.predict(X_val_cv)

        train_acc = get_rmse(y_train_cv, y_train_cv_pred)
        val_acc = get_rmse(y_val_cv, y_val_cv_pred)

        train_scores.append(train_acc)
        val_scores.append(val_acc)

    print('%i Folds' % n_folds)
    print('Mean training error = %.3f +/- %.4f' % (np.mean(train_scores), np.std(train_scores)))
    print('Mean validation error = %.3f +/- %.4f' % (np.mean(val_scores), np.std(val_scores)))
    
    training_rmse.append(np.mean(train_scores))
    training_std.append(np.std(train_scores))
    validation_rmse.append(np.mean(val_scores))
    validation_std.append(np.std(val_scores))
    

    return {'train' : train_scores,
           'val' : val_scores}

#Perform grid search with cross validation to determine optimal hyperparameters
max_depth = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
min_samples_leaf = [1,2,3,4,5,6,7,8,9,10]
min_weight_fraction_leaf = [0,0.1,0.2,0.3,0.4,0.5]

n_folds = 10

grid_search_dict = {}
for j in max_depth:
    grid_search_dict[j] = {}
    for k in min_samples_leaf:
        grid_search_dict[j][k] = {}
        for l in min_weight_fraction_leaf:
            grid_search_dict[j][k][l] = {}
            training_rmse, training_std = [], []
            validation_rmse, validation_std = [], []
            model = DecisionTreeRegressor(max_depth=j,
                                            min_samples_leaf=k,
                                            min_weight_fraction_leaf=l,
                                            random_state=seed)
            #cv_scores = run_cv(n_folds=n_folds,
            #                    model=model,
            #                    X_train=X_train_scaled,
            #                    y_train=y_train)
            #grid_search_dict[j][k][l] = cv_scores