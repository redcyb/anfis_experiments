import numpy
from skfuzzy import gaussmf

from anfis import ANFIS
from membership.membership_functions import MemFuncs
from utils import plotMFs
from datetime import datetime
from utils import plotMFs

ts = numpy.loadtxt("trainingDebug.dat", usecols=[0, 1, 2, 3, 4])

X = ts[:, 0:2]
Y = ts[:, 4]


# mf_gauss = [
#     [
#         {'mean': 4.3, 'sigma': 0.4},
#         {'mean': 5.5, 'sigma': 0.4},
#         {'mean': 6.7, 'sigma': 0.4},
#         {'mean': 7.9, 'sigma': 0.4},
#     ],
#     [
#         {'mean': 2.0, 'sigma': 0.3},
#         {'mean': 2.8, 'sigma': 0.3},
#         {'mean': 3.6, 'sigma': 0.3},
#         {'mean': 4.4, 'sigma': 0.3},
#     ],
#     [
#         {'mean': 1.0, 'sigma': 0.7},
#         {'mean': 2.9, 'sigma': 0.7},
#         {'mean': 4.8, 'sigma': 0.7},
#         {'mean': 6.9, 'sigma': 0.7},
#     ],
#     [
#         {'mean': 0.1, 'sigma': 0.3},
#         {'mean': 0.9, 'sigma': 0.3},
#         {'mean': 1.7, 'sigma': 0.3},
#         {'mean': 2.5, 'sigma': 0.3},
#     ],
# ]

mf_gauss = [
    [
        {'mean': 4.3, 'sigma': 0.7644},
        # {'mean': 6.1, 'sigma': 0.7644},
        {'mean': 7.9, 'sigma': 0.7644},
    ],
    [
        {'mean': 2.0, 'sigma': 0.5096},
        # {'mean': 3.2, 'sigma': 0.5096},
        {'mean': 4.4, 'sigma': 0.5096},
    ],
    # [
    #     {'mean': 1.0, 'sigma': 1.253},
    #     {'mean': 3.95, 'sigma': 1.253},
    #     {'mean': 6.9, 'sigma': 1.253},
    # ],
    # [
    #     {'mean': 0.1, 'sigma': 0.5096},
    #     {'mean': 1.3, 'sigma': 0.5096},
    #     {'mean': 2.5, 'sigma': 0.5096},
    # ],
]

# ===== Show me initial MFS ================================

# varss = [(4.3, 7.9), (2.0, 4.4), (1.0, 6.9), (0.1, 2.5)]
# plotMFs(gaussmf, varss, mf_gauss)


# ===== Learn ANFIS with initial MFs and training data =====

def test(epochs=10):

    mfc = MemFuncs(gaussmf, mf_gauss)

    anf = ANFIS(X, Y, mfc)

    t_start = datetime.now()

    anf.trainHybridJangOffLine(epochs=epochs)

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

test(20)
