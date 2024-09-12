"""Decision tree model fitting with structural paramters"""

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
seed = np.random.seed(18)
plt.rcParams["font.family"] = "Times New Roman"

#Load datasets
df = pd.read_json('data/structural-information-and-frequency-dataset.json')

feature_df = df[['bond angle variance','distortion index','average bond length']]
target_df = df[['target']]

feature_matrix = feature_df.to_numpy()
target = target_df.to_numpy()

#Split data into test and train sets
X_train, X_test, y_train, y_test = train_test_split(feature_matrix, target, test_size=0.25, random_state=10)

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
plt.text(0.6,-1, "$R^2$ = {:.4f}".format(r_square),fontsize=20)
plt.show()

#with open('struc_test_depth6.json', 'w') as file:
#    json.dump(y_test.tolist(), file)
#with open('struc_prediction_depth6.json', 'w') as file:
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
plt.text(-.3500,-.40000, "$R^2$ = {:.4f}".format(r_square),fontsize=20)
plt.show()

#tree.plot_tree(optimized_model, filled=True)
#plt.rcParams['figure.figsize']=[5,5]
#plt.show()

#r = export_text(optimized_model, feature_names=['gm1','gm3_a0','gm3_0a','gm5_a00','gm5_0a0','gm5_00a'])
#with open("output.txt", "a") as f:
#  print(r, file=f)

feat_impt = optimized_model.feature_importances_
features = ['BAV','DI','ABL']

plt.bar(features,feat_impt)
plt.ylabel('Feature Importance',fontsize=15)
plt.show()

feat_impt1 = permutation_importance(optimized_model,X_train_scaled,y_train,random_state=4,scoring='neg_root_mean_squared_error')
mean_importances = feat_impt1.importances_mean
std_dev_importances = feat_impt1.importances_std

plt.bar(features,mean_importances,width=0.6)
plt.ylabel('Feature Importance',fontsize=20)
plt.xticks(fontsize=15,rotation=45)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()


#with open('structure-features.json', 'w') as file:
#    json.dump(features, file)
#with open('structure-feature-importances.json', 'w') as file:
#    json.dump(mean_importances.tolist(), file)

"""SHAP feature importance"""
explainer = shap.Explainer(optimized_model)
shap_values = explainer.shap_values(X_test_scaled)

shap.summary_plot(shap_values, X_test_scaled)

"""Compling all feature importance data to make plot"""

with open('kto-feature-importance-data/irrep-feature-importances.json', 'r') as file:
    irrep_feat_imp = json.load(file)
with open('kto-feature-importance-data/irrep-features.json', 'r') as file:
    irrep_features = json.load(file)
with open('kto-feature-importance-data/structure-feature-importances.json', 'r') as file:
    struc_feat_imp = json.load(file)
with open('kto-feature-importance-data/structure-features.json', 'r') as file:
    struc_features = json.load(file)
with open('kto-feature-importance-data/strain-feature-importances.json', 'r') as file:
    strain_feat_imp = json.load(file)
with open('kto-feature-importance-data/strain-features.json', 'r') as file:
    strain_features = json.load(file)

fig, axs = plt.subplots(1, 3, figsize=(7, 3), gridspec_kw={'width_ratios': [1, 1, 0.6]})

# Plot data on the first subplot
axs[0].bar(irrep_features, irrep_feat_imp, width=0.5)
axs[0].set_ylim(0,8000)
axs[0].set_ylabel('Feature Importance (RMSE)',fontsize=16)

# Plot data on the second subplot
axs[1].bar(strain_features, strain_feat_imp, width=0.5,color='red')
axs[1].set_ylim(0,8000)
struc_features=['BAV','DI','L']
# Plot data on the third subplot
axs[2].bar(struc_features, struc_feat_imp, width=0.5,color='green')
axs[2].set_ylim(0,8000)

axs[0].tick_params(axis='y', labelsize=16)
axs[0].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='x', labelsize=14)
axs[2].tick_params(axis='x', labelsize=14)

axs[1].set_yticks([])
axs[2].set_yticks([])
# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

df1 = pd.DataFrame()
#df1['irrep order parameter features'] = irrep_features
#df1['irrep order parameter feature importances (RMSE)'] = irrep_feat_imp
#df1['strain tensor component features'] = strain_features
#df1['strain tensor component feature importances (RMSE)'] = strain_feat_imp
df1['octahedron structure metric features'] = struc_features
df1['octahedron structure metric feature importances (RMSE)'] = struc_feat_imp
#df1.to_csv('feature-importances-for-structure-model.csv')

"""Compiling all model prediction data for parity plots"""

with open('kto-model-prediction-data/irrep_prediction_depth6.json', 'r') as file:
    irrep_pred = np.array(json.load(file))
with open('kto-model-prediction-data/irrep_test_depth6.json', 'r') as file:
    irrep_test = np.array(json.load(file))
with open('kto-model-prediction-data/struc_prediction_depth10.json', 'r') as file:
    struc_pred = np.array(json.load(file))
with open('kto-model-prediction-data/struc_test_depth10.json', 'r') as file:
    struc_test = np.array(json.load(file))
with open('kto-model-prediction-data/strain_prediction_depth6.json', 'r') as file:
    strain_pred = np.array(json.load(file))
with open('kto-model-prediction-data/strain_test_depth6.json', 'r') as file:
    strain_test = np.array(json.load(file))

fig, axs = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1, 1]})

# Plot data on the first subplot
axs[0].plot(np.linspace(-6.5,6.5,100),np.linspace(-6.5,6.5,100),color='black')
axs[0].plot(irrep_test/10000, irrep_pred/10000, linestyle='none',marker='o')
axs[0].set_ylim(-1.5,1.7)
axs[0].set_xlim(-1.5,1.7)
axs[0].set_ylabel('Predicted Value (10$^{-4}$cm$^{-2}$)', fontsize=20)
axs[0].set_xlabel('DFT Calculated Value (10$^{-4}$cm$^{-2}$)', fontsize=20)
axs[0].text(0.6,-1.1, 'R$^{2}=$0.999' ,fontsize=20)
axs[0].text(-1.3,1.4, 'Irrep Order Parameters' ,fontsize=20)
axs[0].text(-1.3,1.15, 'Tree depth = 6' ,fontsize=20)

# Plot data on the second subplot
axs[1].plot(np.linspace(-6.5,6.5,100),np.linspace(-6.5,6.5,100),color='black')
axs[1].plot(strain_pred/10000, strain_test/10000, linestyle='none',marker='o',color='red')
axs[1].set_ylim(-1.5,1.7)
axs[1].set_xlim(-1.5,1.7)
axs[1].set_xlabel('DFT Calculated Value (10$^{-4}$cm$^{-2}$)', fontsize=20)
axs[1].text(0.6,-1.1, 'R$^{2}=$0.999' ,fontsize=20)
axs[1].text(-1.3,1.4, 'Strain Tensor Components' ,fontsize=20)
axs[1].text(-1.3,1.15, 'Tree depth = 6' ,fontsize=20)

# Plot data on the third subplot
axs[2].plot(np.linspace(-6.5,6.5,100),np.linspace(-6.5,6.5,100),color='black')
axs[2].plot(struc_pred/10000, struc_test/10000, linestyle='none',marker='o',color='green')
axs[2].set_ylim(-1.5,1.7)
axs[2].set_xlim(-1.5,1.7)
axs[2].set_xlabel('DFT Calculated Value (10$^{-4}$cm$^{-2}$)', fontsize=20)
axs[2].text(0.6,-1.1, 'R$^{2}=$0.995' ,fontsize=20)
axs[2].text(-1.3,1.4, 'Octahedron Structure Metrics' ,fontsize=20)
axs[2].text(-1.3,1.15, 'Tree depth = 10' ,fontsize=20)

axs[0].tick_params(axis='y', labelsize=20)
axs[0].tick_params(axis='x', labelsize=20)
axs[1].tick_params(axis='x', labelsize=20)
axs[2].tick_params(axis='x', labelsize=20)

axs[1].set_yticks([])
axs[2].set_yticks([])
# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

with open('kto-model-prediction-data/irrep_prediction_depth6.json', 'r') as file:
    irrep_pred_6 = np.array(json.load(file))
with open('kto-model-prediction-data/irrep_test_depth6.json', 'r') as file:
    irrep_test_6 = np.array(json.load(file))
with open('kto-model-prediction-data/struc_prediction_depth6.json', 'r') as file:
    struc_pred_6 = np.array(json.load(file))
with open('kto-model-prediction-data/struc_test_depth6.json', 'r') as file:
    struc_test_6 = np.array(json.load(file))
with open('kto-model-prediction-data/strain_prediction_depth6.json', 'r') as file:
    strain_pred_6 = np.array(json.load(file))
with open('kto-model-prediction-data/strain_test_depth6.json', 'r') as file:
    strain_test_6 = np.array(json.load(file))
with open('kto-model-prediction-data/irrep_prediction_depth10.json', 'r') as file:
    irrep_pred_10 = np.array(json.load(file))
with open('kto-model-prediction-data/irrep_test_depth10.json', 'r') as file:
    irrep_test_10 = np.array(json.load(file))
with open('kto-model-prediction-data/struc_prediction_depth10.json', 'r') as file:
    struc_pred_10 = np.array(json.load(file))
with open('kto-model-prediction-data/struc_test_depth10.json', 'r') as file:
    struc_test_10 = np.array(json.load(file))
with open('kto-model-prediction-data/strain_prediction_depth10.json', 'r') as file:
    strain_pred_10 = np.array(json.load(file))
with open('kto-model-prediction-data/strain_test_depth10.json', 'r') as file:
    strain_test_10 = np.array(json.load(file))

df = pd.DataFrame()
df['order parameter model prediction with depth 6'] = irrep_pred_6
df['order parameter model dft value with depth 6'] = irrep_test_6
df['strain model prediction with depth 6'] = strain_pred_6
df['strain model dft value with depth 6'] = strain_test_6
df['structure metric model prediction with depth 6'] = struc_pred_6
df['structure metric model prediction with depth 6'] = struc_test_6
df['order parameter model prediction with depth 10'] = irrep_pred_10
df['order parameter model dft value with depth 10'] = irrep_test_10
df['strain model prediction with depth 10'] = strain_pred_10
df['strain model dft value with depth 10'] = strain_test_10
df['structure metric model prediction with depth 10'] = struc_pred_10
df['structure metric model prediction with depth 10'] = struc_test_10

#df.to_csv('parity-plot-data-for-models-with-tree-depths-of-6-and-10.csv')