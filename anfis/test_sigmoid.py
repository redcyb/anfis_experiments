import numpy

from datetime import datetime
from anfis import ANFIS
from membership.membershipfunction import MemFuncs
from utils import plotMFs
from skfuzzy import sigmf

ts = numpy.loadtxt("trainingSetOutTrain.dat", usecols=[0, 1, 2, 3, 4])

X = ts[:, 0:4]
Y = ts[:, 4]

# mf_sigmoids = [
#     [
#         {'b': 5.2, 'c': 8},
#         {'b': 5.8, 'c': 8},
#         {'b': 6.4, 'c': 8},
#         {'b': 7.1, 'c': 8},
#     ],
#     [
#         {'b': 2.5, 'c': 15},
#         {'b': 3, 'c': 15},
#         {'b': 3.5, 'c': 15},
#         {'b': 4, 'c': 15},
#     ],
#     [
#         {'b': 1.7, 'c': 10},
#         {'b': 3.3, 'c': 10},
#         {'b': 4.9, 'c': 10},
#         {'b': 6.3, 'c': 10},
#     ],
#     [
#         {'b': 0.7, 'c': 10},
#         {'b': 1.1, 'c': 10},
#         {'b': 1.5, 'c': 10},
#         {'b': 1.9, 'c': 10},
#     ],
# ]

# mf_sigmoids = [
#     [
#         {'b': 5.2, 'c': 8},
#         {'b': 6.15, 'c': 8},
#         {'b': 7.1, 'c': 8},
#     ],
#     [
#         {'b': 2.5, 'c': 15},
#         {'b': 3.25, 'c': 15},
#         {'b': 4, 'c': 15},
#     ],
#     [
#         {'b': 1.7, 'c': 10},
#         {'b': 4, 'c': 10},
#         {'b': 6.3, 'c': 10},
#     ],
#     [
#         {'b': 0.7, 'c': 10},
#         {'b': 1.3, 'c': 10},
#         {'b': 1.9, 'c': 10},
#     ],
# ]


mf_sigmoids = [
    [
        {'b': 4, 'c': 8},
        {'b': 6, 'c': 8},
        {'b': 8, 'c': 8},
    ],
    [
        {'b': 1.8, 'c': 15},
        {'b': 3.25, 'c': 15},
        {'b': 4.8, 'c': 15},
    ],
    [
        {'b': 1.7, 'c': 10},
        {'b': 4, 'c': 10},
        {'b': 6.3, 'c': 10},
    ],
    [
        {'b': 0.7, 'c': 10},
        {'b': 1.3, 'c': 10},
        {'b': 1.9, 'c': 10},
    ],
]


# ===== Show me initial MFS ================================

def show_mfs():
    varss = [(4.3, 7.9), (2.0, 4.4), (1.0, 6.9), (0.1, 2.5)]
    plotMFs(sigmf, varss, mf_sigmoids)

# show_mfs()


# ===== Learn ANFIS with initial MFs and training data =====

def test():

    mfc = MemFuncs(sigmf, mf_sigmoids)
    anf = ANFIS(X, Y, mfc)

    t_start = datetime.now()

    anf.trainHybridJangOffLine(epochs=10)

    print(anf.consequents[-1][0])
    print(anf.consequents[-2][0])
    print(anf.fitted_values[9][0])

    # if round(anf.consequents[-1][0], 6) == -5.275538 and round(anf.consequents[-2][0], 6) == -1.990703 and round(
    #         anf.fittedValues[9][0], 6) == 0.002249:
    #     print('test is good')

    t_fin = datetime.now()

    print(f"TIME SPENT: {(t_fin - t_start).seconds}s")

    anf.plotErrors()
    anf.plotResults()

# ===== Run ANFIS with test data =====

test()
