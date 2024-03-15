from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import xgboost as xgb
from sklearn.svm import SVR
import joblib
from data_process import read_intensity_data
from sklearn.preprocessing import MinMaxScaler


def model_pred(file_name, PGA, model_type="rf",
               data_file="../Data/intensity_measures_all/intensity_measures_cleaned-0.7.npy"):
    # 获取模型
    model = joblib.load(
        "../Data/save_model/" + model_type + "_" + file_name + "-" + str(PGA) + ".pkl")
    features_list = ['PGD', 'RMSA', 'I_A',
                     'VSI', 'SMA', 'SMV', 'Iv',
                     'Sv(T1)', 'T90', 'FFT']
    # 获取反归一化的值
    x_data = read_intensity_data(features_list, "../Data/datasets/quakeIntensityMeasures" + str(PGA) + ".mat")
    y_data = loadmat("../Data/datasets/" + file_name + "-" + str(PGA) + ".mat")["data"]
    y_data = np.transpose(y_data)
    x_data = np.log(x_data)
    y_data = np.log(y_data)
    x_scaler = MinMaxScaler(feature_range=(0.01, 1))
    y_scaler = MinMaxScaler(feature_range=(0.01, 1))
    x_scaler = x_scaler.fit(x_data)
    y_scaler = y_scaler.fit(y_data)

    del x_data, y_data
    data = np.load(data_file)
    data = np.log(data)
    data = x_scaler.transform(data)
    data_pred = model.predict(data)
    data_pred = np.expand_dims(data_pred, axis=-1)
    data_pred = y_scaler.inverse_transform(data_pred)
    data_pred = np.squeeze(data_pred)

    np.save(
        "../Data/demand_measures_all/demand_measures-" + file_name + "-" + str(PGA) + ".npy",
        data_pred)
    return data_pred


def rf_model_fit(model_file, model_file_params, file_name, PGA):
    """
    Use all the data and the best hyperparameters to fit model, and save.
    :param model_file:
    :param model_file_params:
    :return:
    """
    params = np.load(model_file_params, allow_pickle=True).item()
    n_estimators = params["n_estimators"]
    min_samples_split = params["min_samples_split"]
    max_features = params["max_features"]
    max_depth = params["max_depth"]
    model = RandomForestRegressor(n_estimators=int(n_estimators),
                                  min_samples_split=int(min_samples_split),
                                  max_features=min(max_features, 0.999),
                                  max_depth=int(max_depth),
                                  random_state=2,
                                  n_jobs=-1)

    y_data = loadmat("../Data/datasets/" + file_name + "-" + str(PGA) + ".mat")["data"]
    y_data = np.transpose(y_data)
    y_data = np.log(y_data)
    y_scaler = MinMaxScaler(feature_range=(0.01, 1))
    y_scaler = y_scaler.fit(y_data)

    dataset = loadmat(data_dir + file_name + "-" + str(PGA) + ".mat")
    x_data = dataset["x"]
    y_data = dataset["y"].squeeze()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, test_size=0.3, shuffle=True)
    model = model.fit(x_train, y_train)
    data_pred = model.predict(x_test)

    data_pred = np.expand_dims(data_pred, -1)
    y_test = np.expand_dims(y_test, -1)
    data_pred = y_scaler.inverse_transform(data_pred)
    y_test = y_scaler.inverse_transform(y_test)
    r2 = r2_score(data_pred, y_test)
    rmse = mean_absolute_error(data_pred, y_test)
    print(r2, rmse)
    np.savetxt("../Data/model_valid/rf_" + file_name + "-" + str(PGA) + "-pred.txt", data_pred)
    np.savetxt("../Data/model_valid/rf_" + file_name + "-" + str(PGA) + "-real.txt", y_test)
    joblib.dump(model, model_file)
    return


def svm_model_fit(model_file, model_file_params, file_name, PGA):
    """
    Use all the data and the best hyperparameters to fit model, and save.
    :param model_file:
    :param model_file_params:
    :return:
    """
    params = np.load(model_file_params, allow_pickle=True).item()
    c = params["c"]
    gamma = params["gamma"]
    model = SVR(C=c, gamma=gamma)

    y_data = loadmat("../Data/datasets/" + file_name + "-" + str(PGA) + ".mat")["data"]
    y_data = np.transpose(y_data)
    y_data = np.log(y_data)
    y_scaler = MinMaxScaler(feature_range=(0.01, 1))
    y_scaler = y_scaler.fit(y_data)

    dataset = loadmat(data_dir + file_name + "-" + str(PGA) + ".mat")
    x_data = dataset["x"]
    y_data = dataset["y"].squeeze()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, test_size=0.3, shuffle=True)
    model = model.fit(x_train, y_train)
    data_pred = model.predict(x_test)

    data_pred = np.expand_dims(data_pred, -1)
    y_test = np.expand_dims(y_test, -1)
    data_pred = y_scaler.inverse_transform(data_pred)
    y_test = y_scaler.inverse_transform(y_test)
    r2 = r2_score(data_pred, y_test)
    rmse = mean_absolute_error(data_pred, y_test)
    print(r2, rmse)
    np.savetxt("../Data/model_valid/svm_" + file_name + "-" + str(PGA) + "-pred.txt",
               data_pred)
    np.savetxt("../Data/model_valid/svm_" + file_name + "-" + str(PGA) + "-real.txt", y_test)
    joblib.dump(model, model_file)
    return


def xgb_model_fit(model_file, model_file_params, file_name, PGA):
    """
    Use all the data and the best hyperparameters to fit model, and save.
    :param model_file:
    :param model_file_params:
    :return:
    """
    params = np.load(model_file_params, allow_pickle=True).item()
    n_estimators = params["n_estimators"]
    max_depth = params["max_depth"]
    model = xgb.XGBRegressor(n_estimators=int(n_estimators),  # Count of base learner
                             max_depth=int(max_depth),
                             random_state=2,
                             n_jobs=-1)

    y_data = loadmat("../Data/datasets/" + file_name + "-" + str(PGA) + ".mat")["data"]
    y_data = np.transpose(y_data)
    y_data = np.log(y_data)
    y_scaler = MinMaxScaler(feature_range=(0.01, 1))
    y_scaler = y_scaler.fit(y_data)

    dataset = loadmat(data_dir + file_name + "-" + str(PGA) + ".mat")
    x_data = dataset["x"]
    y_data = dataset["y"].squeeze()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, test_size=0.3, shuffle=True)
    model = model.fit(x_train, y_train)
    data_pred = model.predict(x_test)

    data_pred = np.expand_dims(data_pred, -1)
    y_test = np.expand_dims(y_test, -1)
    data_pred = y_scaler.inverse_transform(data_pred)
    y_test = y_scaler.inverse_transform(y_test)
    r2 = r2_score(data_pred, y_test)
    rmse = mean_absolute_error(data_pred, y_test)
    print(r2, rmse)
    np.savetxt("../Data/model_valid/xgb_" + file_name + "-" + str(PGA) + "-pred.txt",
               data_pred)
    np.savetxt("../Data/model_valid/xgb_" + file_name + "-" + str(PGA) + "-real.txt", y_test)
    joblib.dump(model, model_file)
    return


def im_clean():
    """
    This function can delete the nan value in intensity measures
    """
    data = np.load("../Data/intensity_measures_all/intensity_measures-4.0.npy")
    delete_temp = []
    for i in range(len(data)):
        if data[i, 0] != data[i, 0]:
            delete_temp.append(i)
    data = np.delete(data, delete_temp, axis=0)
    np.save("../Data/intensity_measures_all/intensity_measures_cleaned-4.0.npy", data)


if __name__ == "__main__":
    model_dir = "../Data/save_model/"
    params_dir = "../Data/save_model_params/"
    data_dir = "../Data/datasets/mlDataset"
    model_pred("node", 0.7, "rf", "../Data/intensity_measures_all/intensity_measures_cleaned-0.7.npy")
    model_pred("node", 4.0, "rf", "../Data/intensity_measures_all/intensity_measures_cleaned-0.7.npy")
    # im_clean()
    # rf_model_fit(model_dir + "rf_node-0.7.pkl",
    #              params_dir + "rf_params_node-0.7.npy",
    #              "node", 0.7)
    # rf_model_fit(model_dir + "rf_node-4.0.pkl",
    #              params_dir + "rf_params_node-0.7.npy",
    #              "node", 4.0)
    # svm_model_fit(model_dir + "svm_node-0.7.pkl",
    #               params_dir + "svm_params_node-0.7.npy",
    #               "node", 0.7)
    # svm_model_fit(model_dir + "svm_node-4.0.pkl",
    #               params_dir + "svm_params_node-4.0.npy",
    #               "node", 4.0)
    # xgb_model_fit(model_dir + "xgb_node-0.7.pkl",
    #               params_dir + "xgb_params_node-0.7.npy",
    #               "node", 0.7)
    # xgb_model_fit(model_dir + "xgb_node-4.0.pkl",
    #               params_dir + "xgb_params_node-4.0.npy",
    #               "node", 4.0)
