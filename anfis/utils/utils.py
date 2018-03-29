import json
import os

import matplotlib.pyplot as plt
import numpy as np

from membership.membership_functions import evaluateMFforVar
from utils.functions import activate


def plotMFs(func, varss, mfs_general, file_name=None):

    for k, mfs in enumerate(mfs_general):
        plt.clf()

        xs = np.arange(*varss[k], 0.01)
        ys_2d = [evaluateMFforVar(func, mfs, x) for x in xs]

        ys_arr = np.array(ys_2d)
        ys_arr = ys_arr.transpose()

        for i in range(ys_arr.shape[0]):
            plt.plot(xs, ys_arr[i, :], label="mf{i}".format(i=i))

        if file_name:
            real_path = os.path.realpath(f"../anfis/data/iris/{file_name}__{k}.png")
            plt.savefig(real_path)
        else:
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


def plot_results_v2(Os, Ys):
    plt.clf()
    plt.plot(range(len(Os)), Os, 'ro', label='trained')
    plt.plot(range(len(Ys)), Ys, 'bo', label='original')
    plt.legend(loc='upper left')
    plt.show()


def plot_func(func, x_range, a, b, show=False):
    plt.clf()

    xs = np.arange(*x_range, 0.01)
    xss = [a * x + b for x in xs]

    ys = [activate(func, x) for x in xss]
    plt.plot(xs, ys)

    if show:
        plt.show()


def plot_funcs(funcs, x_range):
    plt.clf()
    for f in funcs:
        func = f[0]
        a = f[1]
        b = f[2]
        plot_func(func, x_range, a, b)
    plt.show()


def gen_mf_by_range(func, mf_count, x_min, x_max):
    x_len = x_max - x_min
    x_step = x_len / (mf_count - 1)

    k = mf_count * 2 / x_len

    mf_means = [-k * (x_min + (x_step * i)) for i in range(mf_count)]
    mf_sigms = [k for i in range(mf_count)]
    mfs = [[func, mf_sigms[i], mf_means[i]] for i in range(mf_count)]

    # plot_funcs(mfs, [x_min, x_max])

    return [[m[1], m[2]] for m in mfs]
