import itertools
import numpy as np
import copy
import matplotlib.pyplot as plt

from lse import LSE
from membership import mf_derivs
from membership.membership_functions import Layer1


class ANFIS2:
    def __init__(self, X, Y, inputs, rules=None):
        self.X = np.array(copy.copy(X))
        self.Y = np.array(copy.copy(Y))
        self.XLen = len(self.X)

        self.layer1 = Layer1(inputs)

        memFuncsByVariable = [
            [mf_num for mf_num in range(len(self.layer1.inputs[inp_num]["mfs"]))]
            for inp_num in range(len(self.layer1.inputs))
        ]

        self.rules = np.array(rules) if rules else np.array(list(itertools.product(*memFuncsByVariable)))

        self.consequents = np.empty(self.Y.ndim * len(self.rules) * (self.X.shape[1] + 1))
        self.consequents.fill(0)

        self.errors = np.empty(0)

        self.memFuncsHomo = all(len(i) == len(memFuncsByVariable[0]) for i in memFuncsByVariable)

        self.trainingType = 'Not trained yet'

    def trainHybridJangOffLine(self, epochs=5, tolerance=1e-5, initialGamma=1000, k=0.01):

        self.trainingType = 'trainHybridJangOffLine'
        convergence = False
        epoch = 1

        while (epoch < epochs) and (convergence is not True):

            # layer 4: forward pass
            layer_4, weights_l2_sums, weights_l2 = self.forward_pass(self.X)

            # layer five: least squares estimate
            layer_5 = np.array(
                LSE(layer_4, self.Y, initialGamma)
            )

            self.consequents = layer_5
            layer_5 = np.dot(layer_4, layer_5)

            # error
            error = np.sum((self.Y - layer_5.T) ** 2)
            print('current error: ', error)

            average_error = np.average(np.absolute(self.Y - layer_5.T))
            self.errors = np.append(self.errors, error)

            if len(self.errors) != 0:
                if self.errors[len(self.errors) - 1] < tolerance:
                    convergence = True

            # back propagation
            if convergence is not True:
                cols = range(len(self.X[0, :]))
                dE_dAlpha = list(
                    self.backprop(colX, cols, weights_l2_sums, weights_l2, layer_5) for colX in range(self.X.shape[1]))

            if len(self.errors) >= 4:
                if self.errors[-4] > self.errors[-3] > self.errors[-2] > self.errors[-1]:
                    k *= 1.1

            if len(self.errors) >= 5:
                if (self.errors[-1] < self.errors[-2]) and (self.errors[-3] < self.errors[-2]) and (
                            self.errors[-3] < self.errors[-4]) and (self.errors[-5] > self.errors[-4]):
                    k *= 0.9

            # handling of variables with a different number of MFs
            t = []
            for x in range(len(dE_dAlpha)):
                for y in range(len(dE_dAlpha[x])):
                    for z in range(len(dE_dAlpha[x][y])):
                        t.append(dE_dAlpha[x][y][z])

            eta = k / np.abs(np.sum(t))

            if np.isinf(eta):
                eta = k

            # handling of variables with a different number of MFs
            dAlpha = copy.deepcopy(dE_dAlpha)
            if not self.memFuncsHomo:
                for x in range(len(dE_dAlpha)):
                    for y in range(len(dE_dAlpha[x])):
                        for z in range(len(dE_dAlpha[x][y])):
                            dAlpha[x][y][z] = -eta * dE_dAlpha[x][y][z]
            else:
                dAlpha = -eta * np.array(dE_dAlpha)

            # === Update params of mfs ===

            for inp_i in range(len(self.layer1.inputs)):
                for mf_i in range(len(self.layer1.inputs[inp_i])):

                    params_list = sorted(self.layer1.inputs[inp_i]["mfs"][mf_i])

                    for param_num in range(len(params_list)):

                        # Update param

                        self.layer1.inputs[inp_i]["mfs"][mf_i][params_list[param_num]] = (
                            self.layer1.inputs[inp_i]["mfs"][mf_i][params_list[param_num]] +
                            dAlpha[inp_i][mf_i][param_num]
                        )

            # === Update params of mfs end ===

            epoch += 1

        self.fitted_values = self.predict(self.X)
        self.residuals = self.Y - self.fitted_values[:, 0]

        return self.fitted_values

    def plotErrors(self):
        if self.trainingType == 'Not trained yet':
            print(self.trainingType)
        else:
            plt.plot(range(len(self.errors)), self.errors, 'ro', label='errors')
            plt.ylabel('error')
            plt.xlabel('epoch')
            plt.show()

    def plotMF(self, x, input_var):
        for m in range(len(self.layer1.inputs[input_var])):
            y = self.layer1.inputs[input_var]["mfs"][m](x)
            plt.plot(x, y, 'r')
        plt.show()

    def plotResults(self):
        if self.trainingType == 'Not trained yet':
            print(self.trainingType)
        else:
            plt.plot(range(len(self.fitted_values)), self.fitted_values, 'r', label='trained')
            plt.plot(range(len(self.Y)), self.Y, 'b', label='original')
            plt.legend(loc='upper left')
            plt.show()

    def forward_pass(self, Xs):
        layer_4 = np.empty(0, )
        weights_l2 = None
        weights_l2_sums = []
        weights_l3_normalized = None
        count_of_X_samples = Xs.shape[0]

        for i in range(count_of_X_samples):
            sample_set = Xs[i, :]

            # layer 1 : FUZZIFICATION
            layer_1 = self.layer_1_forward_pass(sample_set)

            # layer 2 : AGGREGATION
            layer_2, weights_l2 = self.layer_2_forward_pass(layer_1, weights_l2)

            # layer 3 : NORMALIZATION
            layer_3, weights_l2_sums, weights_l3_normalized = self.layer_3_forward_pass(
                layer_2, weights_l2_sums, weights_l3_normalized)

            # pre-layer 4

            # prep for layer four (bit of a hack)
            sample_set_extended = np.append(sample_set, 1)  # 1 is free member (c) in f(x, y) = ax + by + c
            layer_4_matrix = np.array([wght_n * sample_set_extended for wght_n in layer_3])
            row_holder = np.concatenate(layer_4_matrix)

            layer_4 = np.append(layer_4, row_holder)

        weights_l2 = weights_l2.T
        # weights_l3_normalized = weights_l3_normalized.T

        layer_4 = np.array(np.array_split(layer_4, count_of_X_samples))  # layer 4 becomes list of list, by count of sample_sets

        return layer_4, weights_l2_sums, weights_l2

    def backprop(self, columnX, columns, theWSum, theW, theLayerFive):
        paramGrp = [0] * len(self.layer1.inputs[columnX])
        for MF in range(len(self.layer1.inputs[columnX]["mfs"])):

            parameters = np.empty(len(self.layer1.inputs[columnX]["mfs"][MF]))
            timesThru = 0
            for alpha in sorted(self.layer1.inputs[columnX]["mfs"][MF].keys()):

                bucket3 = np.empty(len(self.X))
                for rowX in range(len(self.X)):
                    varToTest = self.X[rowX, columnX]
                    tmpRow = np.empty(len(self.layer1.inputs))
                    tmpRow.fill(varToTest)

                    bucket2 = np.empty(self.Y.ndim)
                    for colY in range(self.Y.ndim):

                        rulesWithAlpha = np.array(np.where(self.rules[:, columnX] == MF))[0]
                        adjCols = np.delete(columns, columnX)

                        senSit = mf_derivs.partial_dMF(
                            self.X[rowX, columnX],
                            self.layer1.inputs[columnX]["mfs"][MF],
                            alpha)

                        # produces d_ruleOutput/d_parameterWithinMF

                        dW_dAplha = senSit * np.array(
                            [np.prod([self.layer1.evaluate_mf(tmpRow)[c][self.rules[r][c]] for c in adjCols]) for r
                             in rulesWithAlpha])

                        bucket1 = np.empty(len(self.rules[:, 0]))

                        for consequent in range(len(self.rules[:, 0])):
                            fConsequent = np.dot(np.append(self.X[rowX, :], 1.), self.consequents[(
                                (
                                    self.X.shape[
                                        1] + 1) * consequent):(
                                ((
                                     self.X.shape[
                                         1] + 1) * consequent) + (
                                    self.X.shape[
                                        1] + 1)),
                                                                                 colY])
                            acum = 0
                            if consequent in rulesWithAlpha:
                                acum = dW_dAplha[np.where(rulesWithAlpha == consequent)] * theWSum[rowX]

                            acum = acum - theW[consequent, rowX] * np.sum(dW_dAplha)
                            acum = acum / theWSum[rowX] ** 2
                            bucket1[consequent] = fConsequent * acum

                        sum1 = np.sum(bucket1)

                        if self.Y.ndim == 1:
                            bucket2[colY] = sum1 * (self.Y[rowX] - theLayerFive[rowX, colY]) * (-2)
                        else:
                            bucket2[colY] = sum1 * (self.Y[rowX, colY] - theLayerFive[rowX, colY]) * (-2)

                    sum2 = np.sum(bucket2)
                    bucket3[rowX] = sum2

                sum3 = np.sum(bucket3)
                parameters[timesThru] = sum3
                timesThru += 1

            paramGrp[MF] = parameters

        return paramGrp

    def layer_1_forward_pass(self, sample_set):
        return self.layer1.evaluate_mf(sample_set)

    def layer_2_forward_pass(self, layer_1, previous_weights_l2):
        mi_alloc = [
            [
                layer_1[var_i][self.rules[fuzzy_term_i][var_i]]  # get result of fuzzy term * var input
                for var_i in range(self.rules.shape[1])  # for each column in rule. rules.shape[1] = vars len
            ]
            for fuzzy_term_i in range(self.rules.shape[0])  # rules rows.shape[0] = mfs terms count
        ]

        layer_2 = np.array([np.product(x) for x in mi_alloc]).T

        if previous_weights_l2 is None:  # if it's first pass
            weights_l2 = layer_2
        else:
            weights_l2 = np.vstack((previous_weights_l2, layer_2))

        return layer_2, weights_l2

    def layer_3_forward_pass(self, layer_2, weights_l2_sums, weights_l3_normalized):
        weights_l2_sum = np.sum(layer_2)  # float
        weights_l2_sums.append(weights_l2_sum)  # array
        layer_3 = layer_2 / weights_l2_sum

        if weights_l3_normalized is None:  # if it's a first pass
            weights_l3_normalized = layer_3
        else:
            weights_l3_normalized = np.vstack((weights_l3_normalized, layer_3))

        return layer_3, weights_l2_sums, weights_l3_normalized

    def predict(self, varsToTest):
        [layerFour, wSum, w] = self.forward_pass(varsToTest)

        # layer five
        layerFive = np.dot(layerFour, self.consequents)

        return layerFive

    def predict_no_learn(self, varsToTest):
        layer_4, weights_l2_sums, weights_l2 = self.forward_pass(varsToTest)
        layer_5 = np.array(LSE(layer_4, self.Y, 1000))
        layer_5 = np.dot(layer_4, layer_5)
        return layer_5
