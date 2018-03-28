import numpy
import os

from skfuzzy import gaussmf

from anfis1 import ANFIS1
from membership.membership_functions import MemFuncs
from utils.utils import plotMFs
from datetime import datetime

ts = numpy.loadtxt(os.path.realpath("../anfis/data/iris/irisTrain.dat"), usecols=[0, 1, 2, 3, 4])

X = ts[:, 0:4]
Y = ts[:, 4:]

mf_gauss = [

    [{"mean": 4.3, "sigma": 0.764389620259217},
     {"mean": 6.1, "sigma": 0.764389620259217},
     {"mean": 7.9, "sigma": 0.764389620259217}],

    [{"mean": 2.0, "sigma": 0.509593080172811},
     {"mean": 3.2, "sigma": 0.509593080172811},
     {"mean": 4.4, "sigma": 0.509593080172811}],

    [{"mean": 1.0, "sigma": 1.25274965542483},
     {"mean": 3.95, "sigma": 1.25274965542483},
     {"mean": 6.9, "sigma": 1.25274965542483}],

    [{"mean": 0.1, "sigma": 0.509593080172811},
     {"mean": 1.3, "sigma": 0.509593080172811},
     {"mean": 2.5, "sigma": 0.509593080172811}]

]


# ===== Show me initial MFS ================================

# varss = [(4.3, 7.9), (2.0, 4.4), (1.0, 6.9), (0.1, 2.5)]
# plotMFs(gaussmf, varss, mf_gauss)


# ===== Learn ANFIS1 with initial MFs and training data =====

def test(epochs=10):
    mfc = MemFuncs(gaussmf, mf_gauss)

    anf = ANFIS1(X, Y, mfc)

    t_start = datetime.now()
    anf.trainHybridJangOffLine(epochs=epochs)
    t_fin = datetime.now()

    print(f"TIME SPENT: {(t_fin - t_start).seconds}s")

    anf.plotErrors()
    anf.plotResults()


# ===== Run ANFIS1 with test data =====

test(20)
