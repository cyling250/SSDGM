import numpy as np

response_dir = "E:/openSeesModelBuilding/DataGeneration/OpenSees3.3.0-x64.exe/bin/results/"


def read_theta(file_num):
    layer = [37, 28, 19, 1, 2, 46, 55]
    data = []
    relat_data = []
    for i in layer:
        data.append(np.abs(np.loadtxt(response_dir + "node" + str(i) + "-" + str(file_num) + ".out")[:, 1]))
    for i in range(6):
        relat_data.append(data[i + 1] - data[i])
    relat_data = np.array(relat_data)
    relat_data = np.abs(relat_data)
    relat_data = np.max(relat_data, axis=1)
    return relat_data


if __name__ == "__main__":
    for i in range(980, 1000):
        print(read_theta(i))
