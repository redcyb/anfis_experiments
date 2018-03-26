import numpy
import os

from datetime import datetime
from anfis2 import ANFIS2
from membership.membership_functions import MemberFunction, Layer1
from utils.matlab_parser import read_mathlab_anfis_structure
from utils.utils import plot_first_layer, read_anfis_from_json_file

ts = numpy.loadtxt(os.path.realpath("../anfis/data/iris/irisTest.dat"), usecols=[0, 1, 2, 3, 4])

X = ts[:, 0:4]
Y = ts[:, 4]

# settings = read_anfis_from_json_file("../anfis/data/iris/4x3_gaussmf_linear.fis.json")
# settings = read_mathlab_anfis_structure("../anfis/data/iris/4x3_gaussmf_linear___before_training.fis")
settings = read_mathlab_anfis_structure("../anfis/data/iris/4x3_gaussmf_linear___after_training.fis")


# ===== Show me initial MFS ================================

# def show_mfs():
#     inputs = []
#     inputs_settings = settings["inputs"]
#
#     for i in inputs_settings:
#         inpoot = {"range": i["range"], "mfs": []}
#         for k in i["mfs"]:
#             mf = MemberFunction(k[0], k[1])
#             inpoot["mfs"].append(mf)
#         inputs.append(inpoot)
#
#     plot_first_layer(inputs)
#
# show_mfs()


# ===== Learn ANFIS with initial MFs and training data =====

def test(epochs=10):
    inputs = []
    outputs = []

    inputs_settings = settings["inputs"]
    outputs_settings = settings["outputs"]
    rules = [r["connections"] for r in settings["rules"][0]]

    for i in inputs_settings:
        inpoot = {"range": i["range"], "mfs": []}
        for k in i["mfs"]:
            mf = MemberFunction(k[0], k[1])
            inpoot["mfs"].append(mf)
        inputs.append(inpoot)

    for o in outputs_settings:
        for j in o["mfs"]:
            coeffs = j[1]
            coeffs.reverse()
            outputs += coeffs

    anf = ANFIS2(X, Y, inputs=inputs,
                 outputs=outputs,
                 # rules=rules
                 )

    # === TRAINING ===

    # t_start = datetime.now()
    # anf.trainHybridJangOffLine(epochs=epochs)
    # t_fin = datetime.now()
    #
    # print(f"TIME SPENT: {(t_fin - t_start).seconds}s")
    #
    # anf.plotErrors()
    # anf.plotResults()

    # === PREDICTION ===

    xx = numpy.array(X)
    yy = numpy.array(Y)
    predicted = anf.predict_no_learn(xx)

    anf.fitted_values = predicted
    anf.residuals = yy - anf.fitted_values[:, 0]

    errors = numpy.array([predicted[i] - Y[i] for i in range(anf.Y.shape[0])])
    average_error = sum([abs(e) for e in errors[:, 0]]) / errors.shape[0]

    print("average_error: ", average_error)

    anf.plotErrors()
    anf.plotResults()

    # === Layer 5 to file ===

    # with open(os.path.realpath("../anfis/data/iris/irisTrainLayer5Result.dat"), "w") as f:
    #     for d in predicted.T[0]:
    #         f.write("".join(str(d)) + "\n")


# ===== Run ANFIS with test data =====

test(20)
