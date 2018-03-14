import numpy

from datetime import datetime
from anfis import ANFIS
from membership.membershipfunction import MemFuncs, MemberFunction
from utils.matlab_parser import read_mathlab_anfis_structure
from utils.utils import plot_first_layer, read_anfis_settings

# ts = numpy.loadtxt("trainingSetOutTrain.dat", usecols=[0, 1, 2, 3, 4])
#
# X = ts[:, 0:4]
# Y = ts[:, 4]

# settings = read_anfis_settings("../anfis/data/iris/4x3_gaussmf_linear.fis.json")
settings = read_mathlab_anfis_structure("../anfis/data/iris/4x3_sigmmf_linear___before_train.fis")

inputs = []
inputs_settings = settings["inputs"]

for i in inputs_settings:
    inpoot = {"range": i["range"], "mfs": []}
    for k in i["mfs"]:
        mf = MemberFunction(k[0], k[1])
        inpoot["mfs"].append(mf)
    inputs.append(inpoot)


# ===== Show me initial MFS ================================

def show_mfs():
    plot_first_layer(inputs)

show_mfs()


# ===== Learn ANFIS with initial MFs and training data =====

# def test():
#
#     mfc = MemFuncs(sigmf, mf_sigmoids)
#     anf = ANFIS(X, Y, mfc)
#
#     t_start = datetime.now()
#
#     anf.trainHybridJangOffLine(epochs=10)
#
#     print(anf.consequents[-1][0])
#     print(anf.consequents[-2][0])
#     print(anf.fitted_values[9][0])
#
#     # if round(anf.consequents[-1][0], 6) == -5.275538 and round(anf.consequents[-2][0], 6) == -1.990703 and round(
#     #         anf.fittedValues[9][0], 6) == 0.002249:
#     #     print('test is good')
#
#     t_fin = datetime.now()
#
#     print(f"TIME SPENT: {(t_fin - t_start).seconds}s")
#
#     anf.plotErrors()
#     anf.plotResults()

# ===== Run ANFIS with test data =====

# test()
