import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import scipy.signal as signal
from matplotlib import pyplot as plt


def segmented_parsing(mass, stiffness, load, delta_time,
                      damping_ratio=0.05, dpm_0=0, vel_0=0,
                      result_length=0):
    """
    分段解析法计算程序，分段解析法一般适用于单自由度体系的动力响应求解，
    所以仅考虑单自由度情况下的线性分段解析法。
    Parameters
    ----------
    load 荷载;一维列表
    delta_time 时间步长;浮点数
    mass 质量;浮点数
    stiffness 刚度;浮点数
    damping_ratio 阻尼比;浮点数
    dpm_0 初始位移;浮点数
    vel_0 初始速度;浮点数
    result_length 结果长度;整数

    Returns 位移，速度;二维数组，二维数组
    -------

    """
    # 前期数据准备
    # 为了方便代码阅读和减少重复参数所进行的参数代换
    omega_n = np.sqrt(stiffness / mass)
    omega_d = omega_n * np.sqrt(1 - damping_ratio ** 2)
    temp_1 = np.e ** (-damping_ratio * omega_n * delta_time)
    temp_2 = damping_ratio / np.sqrt(1 - damping_ratio ** 2)
    temp_3 = 2 * damping_ratio / (omega_n * delta_time)
    temp_4 = (1 - 2 * damping_ratio ** 2) / (omega_d * delta_time)
    temp_5 = omega_n / np.sqrt(1 - damping_ratio ** 2)
    sin = np.sin(omega_d * delta_time)
    cos = np.cos(omega_d * delta_time)

    # 计算所需参数
    A = temp_1 * (temp_2 * sin + cos)
    B = temp_1 * (sin / omega_d)
    C = 1 / stiffness * (temp_3 + temp_1 * (
            (temp_4 - temp_2) * sin - (1 + temp_3) * cos
    ))
    D = 1 / stiffness * (1 - temp_3 + temp_1 * (
            -temp_4 * sin + temp_3 * cos
    ))
    A_prime = -temp_1 * (temp_5 * sin)
    B_prime = temp_1 * (cos - temp_2 * sin)
    C_prime = 1 / stiffness * (-1 / delta_time + temp_1 * (
            (temp_5 + temp_2 / delta_time) * sin + 1 / delta_time * cos
    ))
    D_prime = 1 / (stiffness * delta_time) * (
            1 - temp_1 * (temp_2 * sin + cos)
    )

    # 处理荷载长度
    if result_length == 0:
        result_length = int(1.2 * len(load))
    load = np.pad(load, (0, result_length - len(load)))  # 荷载数据末端补零

    # 初始化位移数组与速度数组
    dpm = np.zeros(result_length)
    vel = np.zeros(result_length)
    acc = np.zeros(result_length)
    dpm[0] = dpm_0
    vel[0] = vel_0

    # 正式开始迭代
    for i in range(result_length - 1):
        dpm[i + 1] = A * dpm[i] + B * vel[i] + C * load[i] + D * load[i + 1]
        vel[i + 1] = A_prime * dpm[i] + B_prime * vel[i] + C_prime * load[i] + D_prime * load[i + 1]
        acc[i + 1] = -2 * damping_ratio * omega_n * vel[i + 1] - stiffness / mass * dpm[i + 1]

    return dpm, vel, acc


def spectrum_scaler():
    En = [3.136, 6.0956, 10.51344]

    data = pd.read_csv('Z:/STEAD/select.csv')
    data = data.iloc[:, :1000]  # select first 1000 waves

    count = 0
    for column in tqdm(data):
        quake_wave = np.array(data[column].values)[::2]  # Scale earthquake into 50hz
        _, _, acc = segmented_parsing(1, (2 * np.pi / 0.99895) ** 2, quake_wave, 0.02,
                                      damping_ratio=0.05, dpm_0=0, vel_0=0,
                                      result_length=3000)
        sa_record = np.abs(acc).max()
        quake_wave = En[0] / sa_record * quake_wave  # 反应谱匹配调幅
        str1 = 'dataopensees/' + str(count) + '.txt'
        with open(str1, 'w') as fp:
            [fp.write(str(item) + '\n') for item in quake_wave]
            fp.close()
        count += 1


def pga_scaler():
    data = pd.read_csv('Z:/STEAD/select.csv')
    data = data.iloc[:, :500]  # select first 1000 waves
    # pga_list = np.linspace(0.01, 9.8, 1000)

    count = 0
    for column in tqdm(data):
        quake_wave = np.array(data[column].values)[::2]  # Scale earthquake into 50hz

        sa_record = np.abs(quake_wave).max()
        quake_wave = 4.0 / sa_record * quake_wave  # 反应谱匹配调幅
        str1 = 'dataopensees/' + str(count) + '.txt'
        with open(str1, 'w') as fp:
            [fp.write(str(item) + '\n') for item in quake_wave]
            fp.close()
        count += 1


if __name__ == "__main__":
    pga_scaler()
