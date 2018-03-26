from random import shuffle

import numpy
import os

from skfuzzy import gaussmf

from anfis1 import ANFIS
from membership.membership_functions import MemFuncs
from datetime import datetime

from utils.utils import plot_results_v2

training_set = numpy.loadtxt(os.path.realpath("../anfis/data/iris/irisTrain.dat"), usecols=[0, 1, 2, 3, 4])
X = training_set[:, 0:4]
Y = training_set[:, 4:]

test_set = numpy.loadtxt(os.path.realpath("../anfis/data/iris/irisTest.dat"), usecols=[0, 1, 2, 3, 4])
shuffle(test_set)
Xt = test_set[:, 0:4]
Yt = test_set[:, 4:]

mf_gauss = [

    [{"mean": 4.37531540162714, "sigma": 0.854530005717575},
     {"mean": 6.094529067108, "sigma": 0.69878766017976},
     {"mean": 7.88379453491342, "sigma": 0.747991283085191}],

    [{"mean": 1.98816748983381, "sigma": 0.427219866533188},
     {"mean": 3.21018936688146, "sigma": 0.426540985486991},
     {"mean": 4.33249902177964, "sigma": 0.586839428666495}],

    [{"mean": 1.0120021953451, "sigma": 1.26294322619014},
     {"mean": 3.92238353782445, "sigma": 1.18815780229168},
     {"mean": 6.91625281999764, "sigma": 1.18686655972051}],

    [{"mean": 0.122848578940821, "sigma": 0.545324414349361},
     {"mean": 1.22680755138137, "sigma": 0.107652643858273},
     {"mean": 2.5506189332926, "sigma": 0.287395808460947}]

]


# ===== Show me initial MFS ================================

# varss = [(4.3, 7.9), (2.0, 4.4), (1.0, 6.9), (0.1, 2.5)]
# plotMFs(gaussmf, varss, mf_gauss)


# ===== Learn ANFIS with initial MFs and training data =====

def test(epochs=1000, threshold=0.0005, learning_rate=0.01):
    mfc = MemFuncs(gaussmf, mf_gauss)

    anf = ANFIS(X, Y, mfc)

    t_start = datetime.now()

    anf.train_backprop_online(
        training_set,
        epochs=epochs,
        threshold=threshold,
        learning_rate=learning_rate)

    t_fin = datetime.now()

    print(f"TIME SPENT: {(t_fin - t_start).seconds}s")

    anf.plotErrors()

    predicted_train, errors_train = anf.predict_online(X, Y)
    plot_results_v2(predicted_train, Y)

    print("Average Training Error:", sum([abs(e) for e in errors_train]) / len(errors_train))

    predicted_test, errors_test = anf.predict_online(Xt, Yt)
    plot_results_v2(predicted_test, Yt)

    print("Average Test Error:", sum([abs(e) for e in errors_train]) / len(errors_test))


# ===== Run ANFIS with test data =====

test()
