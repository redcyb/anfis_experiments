import numpy
import os

from skfuzzy import gaussmf

from anfis_old import ANFIS
from membership.membership_functions import MemFuncs
from utils.utils import plotMFs
from datetime import datetime

ts = numpy.loadtxt(os.path.realpath("../anfis/data/iris/irisTrain.dat"), usecols=[0, 1, 2, 3, 4])

X = ts[:, 0:4]
Y = ts[:, 4]

# mf_gauss = [
#
#     [{"mean": 4.37531540162714, "sigma": 0.854530005717575},
#      {"mean": 6.094529067108, "sigma": 0.69878766017976},
#      {"mean": 7.88379453491342, "sigma": 0.747991283085191}],
#
#     [{"mean": 1.98816748983381, "sigma": 0.427219866533188},
#      {"mean": 3.21018936688146, "sigma": 0.426540985486991},
#      {"mean": 4.33249902177964, "sigma": 0.586839428666495}],
#
#     [{"mean": 1.0120021953451, "sigma": 1.26294322619014},
#      {"mean": 3.92238353782445, "sigma": 1.18815780229168},
#      {"mean": 6.91625281999764, "sigma": 1.18686655972051}],
#
#     [{"mean": 0.122848578940821, "sigma": 0.545324414349361},
#      {"mean": 1.22680755138137, "sigma": 0.107652643858273},
#      {"mean": 2.5506189332926, "sigma": 0.287395808460947}]
#
# ]

# BEFORE MATLAB TRAINING

mf_gauss = [
    [{"mean": 4.3, "sigma": 0.6},
     {"mean": 6.1, "sigma": 0.6},
     {"mean": 7.9, "sigma": 0.6}],

    [{"mean": 2.0, "sigma": 0.4},
     {"mean": 3.2, "sigma": 0.4},
     {"mean": 4.4, "sigma": 0.4}],

    [{"mean": 1.0, "sigma": 1},
     {"mean": 3.95, "sigma": 1},
     {"mean": 6.9, "sigma": 1}],

    [{"mean": 0.1, "sigma": 0.4},
     {"mean": 1.3, "sigma": 0.4},
     {"mean": 2.5, "sigma": 0.4}]
]


# ===== Show me initial MFS ================================

varss = [(4.3, 7.9), (2.0, 4.4), (1.0, 6.9), (0.1, 2.5)]
plotMFs(gaussmf, varss, mf_gauss, "OLD_4x3_gaussmf_linear___before_training")


# ===== Learn ANFIS with initial MFs and training data =====

def test(epochs=10):
    mfc = MemFuncs(gaussmf, mf_gauss)

    anf = ANFIS(X, Y, mfc)

    t_start = datetime.now()
    anf.trainHybridJangOffLine(epochs=epochs)
    t_fin = datetime.now()

    print(f"TIME SPENT: {(t_fin - t_start).seconds}s")

    anf.plotErrors()
    anf.plotResults()

    plotMFs(gaussmf, varss, anf.memClass.mfs_list, "OLD_4x3_gaussmf_linear___MY_training")


# ===== Run ANFIS1 with test data =====

test(100)
