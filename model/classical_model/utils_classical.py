import csv
from cProfile import label

import pandas as pd
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Descriptors as desc
from rdkit.Chem import AllChem, Crippen, Lipinski
# from rdkit.Chem import rdFingerprintGenerator

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
from skopt import BayesSearchCV
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from joblib import dump, load
import json
import time
import random
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN


def objective(model, x_train, y_train, x_val, y_val):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    tp = ((y_pred == y_val) & (y_pred == 1)).sum().item()
    fp = ((y_pred == 1) & (y_pred != y_val)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    return {'loss': -precision, 'status': STATUS_OK}




def model_trainer_hyperopt(x_train, y_train, x_val, y_val, x_test, y_test, model, model_pred_path, model_name, opt_name):
    model.fit(x_train, y_train)
    y_train_model = model.predict(x_train)
    y_val_model = model.predict(x_val)
    y_test_model = model.predict(x_test)

    train_auc = roc_auc_score(y_train, model.predict_proba(x_train)[:, 1], average='weighted')
    train_acc = accuracy_score(y_train, y_train_model)
    train_confusion_matrix = confusion_matrix(y_train, y_train_model)
    train_classification_report = classification_report(y_train, y_train_model, output_dict=True) #['weighted avg']
    print('___________________________')
    print(f'{model_name}_train_AUC:', train_auc)
    print(f'{model_name}_train_ACC:', train_acc)
    print(f'{model_name}_train_confusion_matrix: \n', train_confusion_matrix)
    print(f'{model_name}_train_classification_report:\n', train_classification_report)

    val_auc = roc_auc_score(y_val, model.predict_proba(x_val)[:, 1], average='weighted')
    val_acc = accuracy_score(y_val, y_val_model)
    val_confusion_matrix = confusion_matrix(y_val, y_val_model)
    val_classification_report = classification_report(y_val, y_val_model, output_dict=True)
    print('___________________________')
    print(f'{model_name}_val_AUC:', val_auc)
    print(f'{model_name}_val_ACC:', val_acc)
    print(f'{model_name}_val_confusion_matrix:\n', val_confusion_matrix)
    print(f'{model_name}_val_classification_report:\n', val_classification_report)

    test_auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1], average='weighted')
    test_acc = accuracy_score(y_test, y_test_model)
    test_confusion_matrix = confusion_matrix(y_test, y_test_model)
    test_classification_report = classification_report(y_test, y_test_model, output_dict=True)
    print('___________________________')
    print(f'{model_name}_test_AUC:', test_auc)
    print(f'{model_name}_test_ACC:', test_acc)
    print(f'{model_name}_test_confusion_matrix:\n', test_confusion_matrix)
    print(f'{model_name}_test_classification_report:\n', test_classification_report)

    metrics = {
        'train_ACC': train_acc,
        'train_AUC': train_auc,
        'train_confusion_matrix': train_confusion_matrix,
        'train_classification_report': train_classification_report,
        'val_ACC': val_acc,
        'val_AUC': val_auc,
        'val_confusion_matrix': val_confusion_matrix,
        'val_classification_report': val_classification_report,
        'test_ACC': test_acc,
        'test_AUC': test_auc,
        'test_confusion_matrix': test_confusion_matrix,
        'test_classification_report': test_classification_report
    }
    with open(model_pred_path + f'/Classification_{model_name}_metrics_{opt_name}.txt', 'w') as file:
        for key, value in metrics.items():
            if key in ['train_confusion_matrix', 'train_classification_report',
                       'val_confusion_matrix', 'val_classification_report',
                       'test_confusion_matrix', 'test_classification_report']:
                file.write(f"{key}:\n{value}\n")
            else:
                file.write(f"{key}: {value}\n")

    np.savetxt(model_pred_path + f'Y_pred_train_{model_name}_{opt_name}.csv', y_train_model, delimiter=',')
    np.savetxt(model_pred_path + f'Y_pred_val_{model_name}_{opt_name}.csv', y_val_model, delimiter=',')
    np.savetxt(model_pred_path + f'Y_pred_test_{model_name}_{opt_name}.csv', y_test_model, delimiter=',')
    dump(model, model_pred_path + f'./Classification_{model_name}_model_{opt_name}.joblib')


def train_model(model_name, model, x_train, y_train, x_val, y_val, x_test, y_test, hyperparams_space, param_use, model_pred_path, max_evals, opt_way, opt_name):
    optimized_model = None
    if opt_way == 'hyperopt':
        start_time = time.time()
        trials = Trials()
        best = fmin(fn=lambda params: objective(model, x_train, y_train, x_val, y_val),  # lambda function definition
                    space=hyperparams_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

        best_params = space_eval(hyperparams_space, best)
        print(f'The optimal parameters obtained after {max_evals} iterations of the {model_name} model are：', best_params)
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 3600
        print(f"Time required to optimize parameters after {max_evals} iterations of {model_name} model：{elapsed_time:.2f} hours")


        with open(model_pred_path + 'best_params.txt', 'a') as file:
            file.write(f'The optimal parameters obtained after {max_evals} iterations of the {model_name} model are: {json.dumps(best_params)}\n')
            file.write(f'Time required to optimize parameters after {max_evals} iterations of {model_name} model：{elapsed_time:.2f} hours\n')

        if model_name == 'RF'or model_name == 'KNN':
            optimized_model = model.set_params(**best_params)
        elif model_name == 'SVC':
            best_params['probability'] = True
            optimized_model = model.set_params(**best_params)
    else:
        if param_use is not None:
            if model_name == 'RF'or model_name == 'KNN':
                optimized_model = model.set_params(**param_use)
            elif model_name == 'SVC':
                param_use['probability'] = True
                optimized_model = model.set_params(**param_use)
        else:
            if model_name == 'RF' or model_name == 'KNN':
                optimized_model = model.set_params()
            elif model_name == 'SVC':
                optimized_model = model.set_params(probability=True)

    model_trainer_hyperopt(x_train, y_train, x_val, y_val, x_test, y_test, optimized_model, model_pred_path, model_name, opt_name)




def classical_process(dataset_split_path, model_pred_path):
    x_train = pd.read_csv(dataset_split_path + 'X_train.csv') #, dtype='float32'
    x_val = pd.read_csv(dataset_split_path + 'X_val.csv')
    x_test = pd.read_csv(dataset_split_path + 'X_test.csv')
    y_train = pd.read_csv(dataset_split_path + 'y_train.csv')
    y_val = pd.read_csv(dataset_split_path + 'y_val.csv')
    y_test = pd.read_csv(dataset_split_path + 'y_test.csv')


    x_train = x_train.iloc[:, :]
    x_val = x_val.iloc[:, :]
    x_test = x_test.iloc[:, :]
    print('x_train_shape:', x_train.shape)
    print('x_val_shape:', x_val.shape)
    print('x_test_shape:', x_test.shape)
    print('y_train_shape:', y_train.shape)
    print('y_val_shape:', y_val.shape)
    print('y_test_shape:', y_test.shape)

    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    y_train = y_train.values.ravel()
    y_val = y_val.values.ravel()
    y_test = y_test.values.ravel()
    dump(scaler, model_pred_path + f'Classification_standard_scaler.joblib')



    opt_way2 = 'hyperopt'
    opt_name2 = 'hyperopt'

    max_evals = 50
    param_RF = None
    param_KNN = None
    param_SVC = None


    model_RF = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    model_KNN = KNeighborsClassifier(n_jobs=-1)
    model_SVC = SVC(class_weight='balanced', random_state=42)



    hyperparams_RF = {
        'n_estimators': hp.choice('n_estimators', range(50, 301, 50)),
        'max_depth': hp.choice('max_depth', range(2, 11, 1)),
        'min_samples_split': hp.choice('min_samples_split', range(2, 11, 1)),
        'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 11, 1)),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', None])
    }

    hyperparams_KNN = {
        'n_neighbors': hp.choice('n_neighbors', range(1, 31, 1)),
        'weights': hp.choice('weights', ['uniform', 'distance']),
        'algorithm': hp.choice('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
    }

    hyperparams_SVC = {
        'C': hp.choice('C', [1, 10, 50, 100]),
        'gamma': hp.choice('gamma', [0.01, 0.001, 0.0001]),
        'kernel': hp.choice('kernel', ['rbf']) #'linear', 'poly',
    }

    train_model('RF', model_RF, x_train, y_train, x_val, y_val, x_test, y_test, hyperparams_RF, param_RF, model_pred_path, max_evals, opt_way2, opt_name2)
    train_model('KNN', model_KNN, x_train, y_train, x_val, y_val, x_test, y_test, hyperparams_KNN, param_KNN, model_pred_path, max_evals, opt_way2, opt_name2)
    train_model('SVC', model_SVC, x_train, y_train, x_val, y_val, x_test, y_test, hyperparams_SVC, param_SVC, model_pred_path, max_evals, opt_way2, opt_name2)


def model_predict(final_pre_result, pred_path, model_name, opt_name):
    data_fingerprints = final_pre_result.iloc[:, 2:]
    scaler = load(pred_path + 'Classification_standard_scaler.joblib')
    model = load(pred_path + f'Classification_{model_name}_model_{opt_name}.joblib')
    x_pre = scaler.transform(data_fingerprints)
    y_pred = model.predict(x_pre)

    df2 = pd.DataFrame(final_pre_result, columns=['canonical_smiles'])
    df2['y_pred'] = y_pred
    df2.to_csv(pred_path + f'Y_pred_NIR_{model_name}.csv', index=False)

    y_pred = np.array(y_pred)
    count_1 = np.sum(y_pred == 1)
    count_0 = np.sum(y_pred == 0)
    print(f'The number of samples predicted by the {model_name} model to be 1 is: {count_1}')
    print(f'The number of samples predicted by the {model_name} model to be 0 is: {count_0}')
    print('---------------------------------')




