"""Decision tree model fitting with uniform strain space dataset"""

import os
import json
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_text
from sklearn.inspection import permutation_importance
from scipy.stats import gaussian_kde
import shap
seed = np.random.seed(10)
plt.rcParams["font.family"] = "Times New Roman"

#Load datasets
df = pd.read_json('data/strains-derived-from-irreps-and-frequency-dataset.json')

feature_df = df[['exx','eyy','ezz','exy','exz','eyz']]
target_df = df[['freq']]

feature_matrix = feature_df.to_numpy()
target = target_df.to_numpy()

#Split data into test and train sets
X_train, X_test, y_train, y_test = train_test_split(feature_matrix, target, test_size=0.25, random_state=9)

#Scale the training set data and transform both the training set and test set
scaler = preprocessing.StandardScaler().fit(X_train)  
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


#Declare functions for performing cross validation and calculating error
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
    
fjson = os.path.join('./', "grid-search-data.json")

def write_json(d,fjson):
        with open(fjson, "w") as f:
                json.dump(d,f)
        return d

#write_json(grid_search_dict, fjson)

#Determine model accuracy with test dataset 
optimized_model = DecisionTreeRegressor(max_depth=6,min_samples_leaf=1,min_weight_fraction_leaf=0,random_state=seed).fit(X_train_scaled,y_train)
y_prediction = optimized_model.predict(X_test_scaled)
r_square = optimized_model.score(X_test_scaled,y_test)

plt.plot(np.linspace(-3.5,1.5,100),np.linspace(-3.5,1.5,100),color='black')
plt.plot(y_test/10000, y_prediction/10000, marker='o', linestyle='none',color='C0')
plt.xlabel('Actual Value (10$^{-4}$cm$^{-2}$)', fontsize=20)
plt.ylabel('Predicted Value (10$^{-4}$cm$^{-2}$)', fontsize=20)
plt.xlim(-1.3,1.3)
plt.ylim(-1.3,1.3)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.text(0.6,-1, "$R^2$ = {:.3f}".format(r_square),fontsize=20)
plt.show()

#with open('strain_test_depth10.json', 'w') as file:
#    json.dump(y_test.tolist(), file)
#with open('strain_prediction_depth10.json', 'w') as file:
#    json.dump(y_prediction.tolist(), file)

xy = np.vstack([np.transpose(y_test),y_prediction])
z = gaussian_kde(xy)(xy)

idx = z.argsort()
y_test_ordered, y_prediction_ordered, z = y_test[idx], y_prediction[idx], z[idx]

fig, ax = plt.subplots()
ax.plot(np.linspace(-3.5,1.5,100),np.linspace(-3.5,1.5,100),linewidth=2,color='black',zorder=1)
ax.scatter(y_test_ordered/10000, y_prediction_ordered/10000, c=z, s=30,zorder=2)
plt.xlabel('Actual Value', fontsize=15)
plt.ylabel('Predicted Value', fontsize=15)
plt.xlim(-1.3,1.3)
plt.ylim(-1.3,1.3)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.text(-.600,-10000, "$R^2$ = {:.3f}".format(r_square),fontsize=20)
plt.show()

#tree.plot_tree(optimized_model, filled=True)
#plt.rcParams['figure.figsize']=[5,5]
#plt.show()

#r = export_text(optimized_model, feature_names=['gm1','gm3_a0','gm3_0a','gm5_a00','gm5_0a0','gm5_00a'])
#with open("output.txt", "a") as f:
#  print(r, file=f)

feat_impt = optimized_model.feature_importances_
features = ['$\epsilon_{xx}$','$\epsilon_{yy}$','$\epsilon_{zz}$','$\epsilon_{xy}$','$\epsilon_{xz}$','$\epsilon_{yz}$']

plt.bar(features,feat_impt)
plt.ylabel('Feature Importance',fontsize=15)
plt.show()

feat_impt1 = permutation_importance(optimized_model,X_train_scaled,y_train,random_state=6,scoring='neg_root_mean_squared_error')
mean_importances = feat_impt1.importances_mean
std_dev_importances = feat_impt1.importances_std

plt.bar(features,mean_importances)
plt.ylabel('Feature Importance',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

#with open('strian-features.json', 'w') as file:
#    json.dump(features, file)
#with open('strain-feature-importances.json', 'w') as file:
#    json.dump(mean_importances.tolist(), file)

"""SHAP feature importance"""
explainer = shap.Explainer(optimized_model)
shap_values = explainer.shap_values(X_test_scaled)

shap.summary_plot(shap_values, X_test_scaled)