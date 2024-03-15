from scipy.io import loadmat
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
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


def corrcoef_fig():
    # features_list = ['PGA', 'PGV', 'PGD', 'RMSA', 'RMSV', 'RMSD', 'I_A', 'I_C', 'SED', 'CAV',
    #                  'ASI', 'VSI', 'HI', 'SMA', 'SMV', 'Ia', 'Id', 'Iv',
    #                  'If', 'Sa(T1)', 'Sv(T1)', 'Sd(T1)', 'T70', 'T90', 'FFT']
    features_list = ['PGD', 'RMSA', 'I_A',
                     'VSI', 'SMA', 'SMV', 'Iv',
                     'Sv(T1)', 'T90', 'FFT']
    data = loadmat("../Data/param_study/quakeIntensityMeasures.mat")
    data = np.squeeze(np.array([data[i] for i in features_list]))
    data = np.log(data)
    data = np.corrcoef(data)
    data = pd.DataFrame(data, index=features_list, columns=features_list)
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 9
    plt.figure(figsize=(4.72440945 * 1.5, 3.93700787 * 1.5))
    sns.heatmap(data=data, square=True, annot=True, fmt=".1f", annot_kws={"size": 6.5})
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig("../Data/correalation.tiff", dpi=600, bbox_inches="tight")
    plt.show()


def params_sort(file_name):
    """
    计算参数贡献度排序的函数，将参数贡献度排序保存到csv文件中。
    :param file_name: 需要计算的文件名称 例如 defobearing-1
    :return:
    """
    print("Processing file " + file_name)
    data = loadmat("../Data/datasets/mlDataset" + file_name + ".mat")
    x_data = np.log(data["x"])
    y_data = np.log(data["y"]).squeeze()

    features_list = np.array(['PGD', 'RMSA', 'I_A',
                              'VSI', 'SMA', 'SMV', 'Iv',
                              'Sv(T1)', 'T90', 'FFT'])

    # Only on target once a time.
    importances_list = []
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, test_size=0.3,
                                                        random_state=4072)

    model = RandomForestRegressor()  # Use default parameters to fit model
    model = model.fit(x_train, y_train)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    key_list = features_list[indices]
    importances = importances[indices]
    # Save importances
    importances_list.append(importances)

    # Screen print
    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, key_list[f], importances[f]))
    pred = model.predict(x_test)
    r2 = r2_score(pred, y_test)
    print(r2)

    importances_list = np.transpose(np.array(importances_list))
    data = pd.DataFrame(importances_list, columns=None, index=key_list)
    data.to_csv("../Data/importance_list_" + file_name + ".csv")


if __name__ == "__main__":
    # corrcoef_fig()
    # params_sort("node-0.7")
    print(np.load("../Data/save_model_params/rf_params_node-0.7.npy", allow_pickle=True))
    print(np.load("../Data/save_model_params/rf_params_node-4.0.npy", allow_pickle=True))
