import json
import os

import matplotlib.pyplot as plt
import numpy as np

from membership.membership_functions import evaluateMFforVar


def plotMFs(func, varss, mfs_general):
    for k, mfs in enumerate(mfs_general):
        print()

        xs = np.arange(*varss[k], 0.01)
        ys_2d = [evaluateMFforVar(func, mfs, x) for x in xs]

        ys_arr = np.array(ys_2d)
        ys_arr = ys_arr.transpose()

        for i in range(ys_arr.shape[0]):
            print(mfs[i])
            plt.plot(xs, ys_arr[i, :], label="mf{i}".format(i=i))

        plt.show()


def plot_first_layer(inputs):
    for inp in inputs:
        print(inp)

        xs = np.arange(*inp["range"], 0.01)
        ys_2d = [[mf.evaluate_mf_for_var(x) for x in xs] for mf in inp["mfs"]]

        ys_arr = np.array(ys_2d)

        for i in range(ys_arr.shape[0]):
            plt.plot(xs, ys_arr[i, :], label="mf{i}".format(i=i))

        plt.show()


def read_anfis_from_json_file(file_name):
    real_path = os.path.realpath(file_name)

    with open(real_path, "r") as f:
        data = f.read()
        data = json.loads(data)

    return data
