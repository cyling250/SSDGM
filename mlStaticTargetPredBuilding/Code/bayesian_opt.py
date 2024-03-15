from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from sklearn.model_selection import KFold
from sklearn.svm import SVR
import warnings


def rf_bayesian_opt(file_name):
    def RF_evaluate(n_estimators, min_samples_split, max_features, max_depth):
        nonlocal x_data, y_data
        val = cross_val_score(
            RandomForestRegressor(n_estimators=int(n_estimators),
                                  min_samples_split=int(min_samples_split),
                                  max_features=min(max_features, 0.999),
                                  max_depth=int(max_depth),
                                  random_state=2,
                                  n_jobs=-1),
            x_data, y_data, scoring='r2', cv=KFold(n_splits=5, shuffle=True)
        ).mean()
        return val

    # Read data
    data = loadmat("../Data/datasets/mlDataset" + file_name + ".mat")
    x_data = data["x"]
    y_data = data["y"].squeeze()

    # Define range of values.
    pbounds = {'n_estimators': (10, 300),  # Range from 50 to 250
               'min_samples_split': (2, 100),
               'max_features': (0.1, 0.999),
               'max_depth': (5, 100)}

    RF_bo = BayesianOptimization(
        f=RF_evaluate,  # Target function
        pbounds=pbounds,  # Range of value
        verbose=2,  # verbose = 2 print all，verbose = 1 print the max，verbose = 0 print nothing
        random_state=1,
    )

    RF_bo.set_gp_params(alpha=1e-5, kernel=None)
    utility = UtilityFunction()
    RF_bo.maximize(acquisition_function=utility, init_points=5, n_iter=100)

    res = RF_bo.max
    params_max = res['params']
    print(params_max)
    np.save("../Data/save_model_params/rf_params_" + file_name + ".npy", params_max)


def svm_bayesian_opt(file_name):
    def svm_evaluate(c, gamma):
        nonlocal x_data, y_data
        val = cross_val_score(
            SVR(C=c, gamma=gamma),
            x_data, y_data, scoring='r2', cv=KFold(n_splits=5, shuffle=True)
        ).mean()
        return val

    # Read data
    data = loadmat("../Data/datasets/mlDataset" + file_name + ".mat")
    x_data = np.log(data["x"])
    y_data = np.log(data["y"]).squeeze()

    # Define range of values.
    pbounds = {"c": (0.2, 10),
               "gamma": (0.1, 100)
               }

    SVM_bo = BayesianOptimization(
        f=svm_evaluate,  # Target function
        pbounds=pbounds,  # Range of value
        verbose=2,  # verbose = 2 print all，verbose = 1 print the max，verbose = 0 print nothing
        random_state=1,
        allow_duplicate_points=True
    )

    SVM_bo.set_gp_params(alpha=1e-5, kernel=None)
    utility = UtilityFunction()
    SVM_bo.maximize(acquisition_function=utility, init_points=5, n_iter=100)

    res = SVM_bo.max
    params_max = res['params']
    print(params_max)
    np.save("../Data/save_model_params/svm_params_" + file_name + ".npy", params_max)


def xgb_bayesian_opt(file_name):
    def xgb_evaluate(n_estimators, max_depth):
        nonlocal x_data, y_data
        val = cross_val_score(
            xgb.XGBRegressor(n_estimators=int(n_estimators),  # Count of base learner
                             max_depth=int(max_depth),
                             random_state=2,
                             n_jobs=-1),
            x_data, y_data, scoring='r2', cv=KFold(n_splits=5, shuffle=True)
        ).mean()
        return val

    # Read data
    data = loadmat("../Data/datasets/mlDataset" + file_name + ".mat")
    x_data = data["x"]
    y_data = data["y"].squeeze()

    # Define range of values.
    pbounds = {'n_estimators': (10, 300),  # Range from 50 to 250
               'max_depth': (5, 100)}

    RF_bo = BayesianOptimization(
        f=xgb_evaluate,  # Target function
        pbounds=pbounds,  # Range of value
        verbose=2,  # verbose = 2 print all，verbose = 1 print the max，verbose = 0 print nothing
        random_state=1,
    )

    RF_bo.set_gp_params(alpha=1e-5, kernel=None)
    utility = UtilityFunction()
    RF_bo.maximize(acquisition_function=utility, init_points=5, n_iter=100)

    res = RF_bo.max
    params_max = res['params']
    print(params_max)
    np.save("../Data/save_model_params/xgb_params_" + file_name + ".npy", params_max)


if __name__ == "__main__":
    # rf_bayesian_opt("node-" + "0.7")
    # rf_bayesian_opt("node-" + "4.0")
    svm_bayesian_opt("node-" + "0.7")
    svm_bayesian_opt("node-" + "4.0")
    xgb_bayesian_opt("node-" + "0.7")
    xgb_bayesian_opt("node-" + "4.0")
