import numpy as np
from scipy.io import savemat, loadmat
from tqdm import tqdm
import warnings
from sklearn.preprocessing import MinMaxScaler
from scipy.fftpack import fft
from matplotlib import pyplot as plt

quake_wave_count = 500
work_dir = "../Data/datasets/"


class IntensityMeasure:
    def __init__(self, quake_waves=None, default_features=None):
        """
        Input the quake_waves with size (batch_size,seq_len)
        Output the normalized intensity measures with size (batch_size,features_num)
        features_list = ['$PGA$', '$PGV$', '$PGD$', '$RMSA$', '$RMSV$', '$RMSD$', '$I_A$',
                         '$ASI$', '$VSI$', '$HI$', '$SMV$', '$Sa(T1)$', '$Sv(T1)$', '$T70$']
        :param quake_waves: ndarray
        """
        super(IntensityMeasure, self).__init__()
        self.time_step = 0.02
        self.batch_size = quake_waves.shape[0]
        self.seq_len = quake_waves.shape[1]
        self.quake_waves_a = quake_waves
        self.quake_waves_v = None
        self.quake_waves_d = None
        if default_features:
            self.default_features = default_features
        else:
            self.default_features = ['PGA', 'PGV', 'PGD', 'RMSA', 'RMSV', 'RMSD', 'I_A', 'I_C', 'SED', 'CAV',
                                     'ASI', 'VSI', 'HI', 'SMA', 'SMV', 'Ia', 'Id', 'Iv',
                                     'If', 'Sa(T1)', 'Sv(T1)', 'Sd(T1)', 'T70', 'T90', 'FFT']
        self.params = {}

    def get_params_array(self):
        """
        Change the params dict into ordered array
        :return:
        """
        if not self.params:
            raise ValueError("There is no param in dict, run 'make_intensity_measure_file' first.")
        else:
            result = np.zeros((self.batch_size, len(self.default_features)))
            for i in range(len(self.default_features)):
                result[:, i] = self.params[self.default_features[i]]
            return result

    def make_intensity_measure_file(self, is_save=False, save_file=None):
        self.params["PGA"] = self.get_pga()
        self.params["PGV"] = self.get_pgv()
        self.params["PGD"] = self.get_pgd()
        self.params["RMSA"] = self.get_rmsa()
        self.params["RMSV"] = self.get_rmsv()
        self.params["RMSD"] = self.get_rmsd()
        self.params["I_A"] = self.get_i_a()
        self.params["I_C"] = self.get_i_c()
        self.params["SED"] = self.get_sed()
        self.params["CAV"] = self.get_cav()
        self.params["ASI"] = self.get_asi()
        self.params["VSI"] = self.get_vsi()
        self.params["HI"] = self.get_hi()
        self.params["SMA"] = self.get_sma()
        self.params["SMV"] = self.get_smv()
        self.params["Ia"] = self.get_ia()
        self.params["Id"] = self.get_id()
        self.params["Iv"] = self.get_iv()
        self.params["If"] = self.get_if()
        self.params["Sa(T1)"] = self.get_s_a()
        self.params["Sv(T1)"] = self.get_s_v()
        self.params["Sd(T1)"] = self.get_s_d()
        self.params["T70"] = self.get_t(0.15, 0.85)
        self.params["T90"] = self.get_t(0.05, 0.95)
        self.params["FFT"] = self.get_fft(-300, -1)
        # TODO Add more intensity measures

        if is_save:
            self.save(save_file)

        result = {}
        for i in self.default_features:
            result[i] = self.params[i]

        return result

    def get_pga(self):
        print("Get pga.")
        return np.abs(self.quake_waves_a).max(1)

    def get_pgv(self):
        print("Get pgv.")
        self.quake_waves_v = np.zeros(self.quake_waves_a.shape)
        for i in range(self.quake_waves_a.shape[1]):
            self.quake_waves_v[:, i] = self.quake_waves_a[:, :i].sum(1) * self.time_step  # Where 0.01 is the time step
        return np.abs(self.quake_waves_v).max(1)

    def get_pgd(self):
        print("Get pgd.")
        self.quake_waves_d = np.zeros(self.quake_waves_a.shape)
        if self.quake_waves_v is None:
            self.get_pgv()
        for i in range(self.quake_waves_a.shape[1]):
            self.quake_waves_d[:, i] = self.quake_waves_v[:, :i].sum(1) * self.time_step
        return np.abs(self.quake_waves_d).max(1)

    def get_rmsa(self):
        print("Get rmsa.")
        return ((self.quake_waves_a ** 2).sum(1) * self.time_step / 60) ** 0.5

    def get_rmsv(self):
        print("Get rmsv.")
        return ((self.quake_waves_v ** 2).sum(1) * self.time_step / 60) ** 0.5

    def get_rmsd(self):
        print("Get rmsd.")
        return ((self.quake_waves_d ** 2).sum(1) * self.time_step / 60) ** 0.5

    def get_i_a(self):
        print("Get i_a.")
        return (self.quake_waves_a ** 2).sum(1) * self.time_step * np.pi / (2 * 9.8)

    def get_i_c(self):
        print("Get i_c.")
        return self.params["RMSA"] ** 1.5 * ((self.time_step * self.quake_waves_a.shape[1]) ** 0.5)

    def get_sed(self):
        print("Get sed.")
        return (self.quake_waves_v ** 2).sum(1) * self.time_step

    def get_cav(self):
        print("Get cav.")
        return np.abs(self.quake_waves_a).sum(1) * self.time_step

    def get_s_a(self):
        print("Get s_a.")
        dpm, vel, acc = self.segmented_parsing()
        return np.abs(acc).max(1)

    def get_s_v(self):
        print("Get s_v.")
        dpm, vel, acc = self.segmented_parsing()
        return np.abs(vel).max(1)

    def get_s_d(self):
        print("Get s_d.")
        dpm, vel, acc = self.segmented_parsing()
        return np.abs(dpm).max(1)

    def get_asi(self):
        print("Get asi.")
        result = np.zeros(self.batch_size)
        for period in tqdm(np.arange(0.1, 0.51, 0.01), position=0):
            dpm, vel, acc = self.segmented_parsing(period)
            result += np.abs(acc).max(1) * 0.01
        return result

    def get_vsi(self):
        print("Get vsi.")
        result = np.zeros(self.batch_size)
        for period in tqdm(np.arange(0.1, 2.51, 0.01), position=0):
            dpm, vel, acc = self.segmented_parsing(period)
            result += np.abs(vel).max(1) * 0.01
        return result

    def get_t(self, start, end):
        label = str(end - start)
        print("Get duration %.2f." % (end - start))
        duration = np.zeros(self.batch_size)
        numerator = np.zeros(self.quake_waves_a.shape)
        denominator = (self.quake_waves_a ** 2).sum(1) * self.time_step
        start = start * denominator
        end = end * denominator
        for i in range(self.seq_len):
            numerator[:, i] = self.quake_waves_a[:, i] ** 2 * self.time_step
        for i in tqdm(range(self.batch_size)):
            is_start = False
            for j in range(self.seq_len):
                if is_start:
                    if numerator[i, :j].sum() > end[i]:
                        duration[i] = j - is_start
                        break
                    continue
                if numerator[i, :j].sum() > start[i]:
                    is_start = j
        # Here, t80 means t90.
        return duration * self.time_step

    def get_hi(self):
        print("Get hi.")
        result = np.zeros(self.batch_size)
        for period in tqdm(np.arange(0.1, 2.51, 0.01)):
            dpm, vel, acc = self.segmented_parsing(period)
            result += np.abs(dpm).max(1) * (2 * np.pi / period) * 0.01
        return result

    def get_sma(self):
        print("Get sma.")
        return np.sort(np.abs(self.quake_waves_a), axis=1)[:, -3]

    def get_smv(self):
        print("Get smv.")
        return np.sort(np.abs(self.quake_waves_v), axis=1)[:, -3]

    def get_miv(self):
        pass

    def get_di(self):
        pass

    def get_ia(self):
        print("Get ia.")
        return self.params["PGA"] * 60 ** (1 / 3)

    def get_id(self):
        print("Get id.")
        return self.params["PGD"] * 60 ** (1 / 3)

    def get_iv(self):
        print("Get iv.")
        return (self.params["PGV"]) ** (2 / 3) * 60 ** (1 / 3)

    def get_if(self):
        print("Get if.")
        return self.params["PGV"] * 60 ** (1 / 4)

    def get_fft(self, start, end):
        print("Get fft.")
        fft_result = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            fft_wave = fft(self.quake_waves_a[i])
            abs_y = np.abs(fft_wave)  # 取复数的绝对值，即复数的模(双边频谱)
            normalization_y = abs_y / self.seq_len  # 归一化处理（双边频谱）
            fft_result[i] = np.sum(normalization_y[range(int(self.seq_len / 2), self.seq_len)][start: end])
        return fft_result

    def save(self, file_name="../Data/datasets/quakeIntensityMeasures.mat"):
        result = {}
        for i in self.default_features:
            result[i] = self.params[i]
        savemat(file_name, result)

    def segmented_parsing(self, period=0.9987):
        """
        Segmental analysis for linear structure response.
        """
        stiffness = (period / (2 * np.pi)) ** 2
        mass = 1
        delta_time = self.time_step
        damping_ratio = 0.05
        dpm_0 = np.zeros(self.batch_size)
        vel_0 = np.zeros(self.batch_size)
        result_length = self.seq_len
        load = self.quake_waves_a
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

        # Initialize Displacement Array and Velocity Array
        dpm = np.zeros((self.batch_size, result_length))
        vel = np.zeros((self.batch_size, result_length))
        acc = np.zeros((self.batch_size, result_length))
        dpm[:, 0] = dpm_0
        vel[:, 0] = vel_0

        for i in range(result_length - 1):
            dpm[:, i + 1] = A * dpm[:, i] + B * vel[:, i] + C * load[:, i] + D * load[:, i + 1]
            vel[:, i + 1] = A_prime * dpm[:, i] + B_prime * vel[:, i] + C_prime * load[:, i] + D_prime * load[:, i + 1]
            acc[:, i + 1] = -2 * damping_ratio * omega_n * vel[:, i + 1] - stiffness / mass * dpm[:, i + 1]

        return dpm, vel, acc


def read_intensity_data(key_list, file_name=work_dir + "quakeIntensityMeasures.mat"):
    """
    Read data into numpy, while the different params are organized as the last feature.
    :return: ndarray
    """
    data = loadmat(file_name)
    feature_count = len(key_list)
    quake_batch = data[key_list[0]].shape[1]
    intensity_measures = np.zeros((quake_batch, feature_count))
    for i in range(len(key_list)):
        intensity_measures[:, i] = data[key_list[i]].squeeze()
    return intensity_measures  # Size (batch_size,features)


def read_quake_wave(start, end, PGA):
    data = []
    for i in tqdm(range(start, end), position=0):
        with open("E:/openSeesModelBuilding/quake_wave" + str(PGA) + "/" + str(i) + ".txt", "r") as fp:
            quake_wave = fp.readlines()
            quake_wave = [j for j in quake_wave]
            quake_wave = np.array(quake_wave)
            quake_wave = quake_wave.astype(np.float32)
        data.append(quake_wave)
    data = np.array(data)
    savemat(work_dir + "quakeWave" + str(PGA) + ".mat",
            {
                "quake_wave": data
            })
    return


def read_param(file_name, PGA, start=0, end=500):
    data = np.zeros(end - start)
    for i in tqdm(range(start, end), position=0):
        data[i - start] = get_max_interlayer_dis_angle(i, PGA)
    savemat(work_dir + file_name + "-" + str(PGA) + ".mat", {
        "data": data  # Size of (1000)
    })


def get_max_interlayer_dis_angle(wave_num, PGA):
    layer_list = [37, 28, 19, 1, 2, 46, 55]
    layer_duration = []
    param_dir = "E:/openSeesModelBuilding/result"
    for node_num in range(len(layer_list)):
        param_file_name = param_dir + str(PGA) + "/node" + str(layer_list[node_num]) + "-" + str(wave_num) + ".out"
        with open(param_file_name, "r") as fp:
            temp_duration = np.array(fp.readlines())
            temp_duration = [list(map(float, j.split(" "))) for j in temp_duration]
            temp_duration = [j[1] for j in temp_duration]
            layer_duration.append(temp_duration)
    for i in range(6):
        layer_duration[i] = np.array(layer_duration[i + 1]) - np.array(layer_duration[i])
    layer_duration = np.array(layer_duration[:-1])
    layer_duration = np.abs(layer_duration)
    layer_duration = layer_duration.max()
    return layer_duration/3000


def get_data_set_ml(file_name, start, end, PGA, features_list=None, is_read_quake=False):
    if is_read_quake:
        print("Reading quake waves from txt to mat.")
        read_quake_wave(start, end, PGA)
        print("Calculating the intensity measures of quake waves.")
        quake_waves = loadmat(work_dir + "/quakeWave" + str(PGA) + ".mat")["quake_wave"]
        quake_waves_params = IntensityMeasure(quake_waves, features_list)
        del quake_waves
        quake_waves_params.make_intensity_measure_file(True,
                                                       "../Data/datasets/quakeIntensityMeasures" + str(PGA) + ".mat")
    key_list = list(loadmat(work_dir + "quakeIntensityMeasures" + str(PGA) + ".mat").keys())[3:]
    print(key_list)
    print("Calculating the demand measures of response.")
    read_param(file_name, PGA, start, end)
    print("Create the data set for machine learning.")
    x_data = read_intensity_data(key_list, work_dir + "quakeIntensityMeasures" + str(PGA) + ".mat")
    y_data = loadmat(work_dir + file_name + "-" + str(PGA) + ".mat")["data"]
    y_data = np.transpose(y_data)

    # logarithmic
    x_data = np.log(x_data)
    y_data = np.log(y_data)

    # Numerical normalization, the purpose is to sort, so the normalization range can be set by yourself
    x_scaler = MinMaxScaler(feature_range=(0.01, 1))
    x_scaler = x_scaler.fit(x_data)
    x_data = x_scaler.transform(x_data)
    y_scaler = MinMaxScaler(feature_range=(0.01, 1))
    y_scaler = y_scaler.fit(y_data)
    y_data = y_scaler.transform(y_data)

    # Log and scaler are in this function-last step
    savemat(work_dir + "mlDataset" + file_name + "-" + str(PGA) + ".mat", {
        "x": x_data,
        "y": y_data
    })
    print("Done.")
    return


if __name__ == "__main__":
    features_list = ['PGD', 'RMSA', 'I_A',
                     'VSI', 'SMA', 'SMV', 'Iv',
                     'Sv(T1)', 'T90', 'FFT']
    get_data_set_ml("node", 0, 500, 0.7, is_read_quake=False, features_list=features_list)
    get_data_set_ml("node", 0, 500, 4.0, is_read_quake=False, features_list=features_list)
