import matplotlib.pyplot as plt
import numpy as np

from membership.membershipfunction import evaluateMFforVar


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
