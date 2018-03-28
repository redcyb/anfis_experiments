import itertools
from random import shuffle

import numpy as np
import copy
import matplotlib.pyplot as plt

from lse import LSE
from membership import mf_derivs
from utils.functions import derivative, activate
from utils.utils import gen_mf_by_range


class ANFIS1:
    layer_1A = None
    layer_2Z = None
    layer_2A = None
    layer_3 = None

    layer_1mfZ = None

    layer_4A = None
    layer_4Z = None

    layer_3to4_A = None
    layer_4_params = None
    layer_5A = None  # Output of layer 5
    layer_5Z = None  # Weighted sum of layer 4 output
    E_out = None
    weights_4to5 = None
    weights_Xto1 = None
    dE_by_A5 = None

    l5_activation_func = None

    predicted = None

    def __init__(self, X, Y, memFunction, l5_activation_func="linear"):
        self.X = np.array(copy.copy(X))
        self.Y = np.array(copy.copy(Y))
        self.XLen = len(self.X)

        self.memFuncsInitialObj = memFunction

        self.memClass = copy.deepcopy(memFunction)
        self.memFuncs = self.memClass.mfs_list
        self.memFuncsByVariable = [[x for x in range(len(self.memFuncs[z]))] for z in range(len(self.memFuncs))]

        self.rules = np.array(list(itertools.product(*self.memFuncsByVariable)))

        self.consequents = np.empty(self.Y.ndim * len(self.rules) * (self.X.shape[1] + 1))
        self.consequents.fill(0)

        self.errors = np.empty(0)

        self.memFuncsHomo = all(len(i) == len(self.memFuncsByVariable[0]) for i in self.memFuncsByVariable)

        self.trainingType = 'Not trained yet'

        self.l5_activation_func = l5_activation_func

        limits = [
            [X[:, i].min(), X[:, i].max()] for i in range(X.shape[1])
            ]

        self.weights_Xto1 = [np.array(gen_mf_by_range("gaussian", 3, *i)) for i in limits]

        pass

    def trainHybridJangOffLine(self, epochs=5, tolerance=1e-5, initialGamma=1000, k=0.01):

        self.trainingType = 'trainHybridJangOffLine'
        convergence = False
        epoch = 1

        while (epoch < epochs) and (convergence is not True):

            # layer 4: forward pass
            layer_4, weights_l2_sums, weights_l2 = self.forward_half_pass(self.X)

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

            for varsWithMemFuncs in range(len(self.memFuncs)):
                for MFs in range(len(self.memFuncsByVariable[varsWithMemFuncs])):
                    paramList = sorted(self.memFuncs[varsWithMemFuncs][MFs])
                    for param in range(len(paramList)):
                        self.memFuncs[varsWithMemFuncs][MFs][paramList[param]] = (
                            self.memFuncs[varsWithMemFuncs][MFs][paramList[param]] +
                            dAlpha[varsWithMemFuncs][MFs][param])
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

    def plotMF(self, x, input_var, func):
        for mf in range(len(self.memFuncs[input_var])):
            y = func(x, **self.memClass.MFList[input_var][mf])
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

    def plot_results_v2(self, Ys):
        plt.plot(range(self.layer_5A[0]), self.layer_5A, 'r', label='trained')
        plt.plot(range(Ys.shape[0]), Ys, 'b', label='original')
        plt.legend(loc='upper left')
        plt.show()

    def forward_half_pass(self, Xs):
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

        layer_4 = np.array(
            np.array_split(layer_4, count_of_X_samples))  # layer 4 becomes list of list, by count of sample_sets

        return layer_4, weights_l2_sums, weights_l2

    def backprop(self, columnX, columns, theWSum, theW, theLayerFive):
        paramGrp = [0] * len(self.memFuncs[columnX])
        for MF in range(len(self.memFuncs[columnX])):

            parameters = np.empty(len(self.memFuncs[columnX][MF]))
            timesThru = 0
            for alpha in sorted(self.memFuncs[columnX][MF].keys()):

                bucket3 = np.empty(len(self.X))
                for rowX in range(len(self.X)):
                    varToTest = self.X[rowX, columnX]
                    tmpRow = np.empty(len(self.memFuncs))
                    tmpRow.fill(varToTest)

                    bucket2 = np.empty(self.Y.ndim)
                    for colY in range(self.Y.ndim):

                        rulesWithAlpha = np.array(np.where(self.rules[:, columnX] == MF))[0]
                        adjCols = np.delete(columns, columnX)

                        senSit = mf_derivs.partial_dMF(
                            self.X[rowX, columnX],
                            self.memFuncs[columnX][MF],
                            alpha,
                            self.memFuncsInitialObj.func)

                        # produces d_ruleOutput/d_parameterWithinMF
                        dW_dAplha = senSit * np.array(
                            [np.prod([self.memClass.evaluate_mf(tmpRow)[c][self.rules[r][c]] for c in adjCols]) for r
                             in rulesWithAlpha])

                        bucket1 = np.empty(len(self.rules[:, 0]))
                        for consequent in range(len(self.rules[:, 0])):
                            fConsequent = np.dot(np.append(self.X[rowX, :], 1.), self.consequents[(
                                (self.X.shape[1] + 1) * consequent):(
                            ((self.X.shape[1] + 1) * consequent) + (self.X.shape[1] + 1)), colY])
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
        return self.memClass.evaluate_mf(sample_set)

    def layer_2_forward_pass(self, layer_1, previous_weights_l2):
        mi_alloc = []
        for fuzzy_term_i in range(self.rules.shape[0]):
            res = []
            for var_i in range(self.rules.shape[1]):
                rr = self.rules[fuzzy_term_i][var_i]
                v1 = layer_1[var_i][rr]
                res.append(v1)
            mi_alloc.append(res)

        # mi_alloc = [
        #     [
        #         layer_1[var_num][self.rules[fuzzy_term][var_num]]  # get result of fuzzy term * var input
        #         for var_num in range(self.rules.shape[1])  # for each column in rule. rules.shape[1] = vars len
        #         ]
        #     for fuzzy_term in range(self.rules.shape[0])  # rules rows.shape[0] = mfs terms count
        #     ]

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
        [layerFour, wSum, w] = self.forward_half_pass(varsToTest)

        # layer five
        layerFive = np.dot(layerFour, self.consequents)

        return layerFive

    # ============ backprop online learning algorithms ===========================================

    def train_backprop_online(self, training_set, epochs=10, threshold=0.001, learning_rate=0.01):

        self.trainingType = 'backprop online'
        # shuffle(training_set)
        Xs = training_set[:, 0:4]
        Ys = training_set[:, 4:]

        epoch = 0

        while epoch < epochs:
            # if not epoch % 100:
            #     shuffle(training_set)
            #     Xs = training_set[:, 0:4]
            #     Ys = training_set[:, 4:]

            epoch += 1
            epoch_errors = []

            for z in range(Xs.shape[0]):
                x = Xs[z]
                y = Ys[z]

                self.forward_online(x)

                self.E_out = np.power((y - self.layer_5A), 2) / 2
                E_out_summ = np.sum(self.E_out, axis=0)
                epoch_errors.append(E_out_summ)

                self.dE_by_A5 = self.layer_5A - y

                self.backprop_online(learning_rate=learning_rate)

            # Service stuff

            epoch_error = sum(epoch_errors) / epochs
            self.errors = np.append(self.errors, epoch_error)

            if not epoch % 10:
                print(f"\nEpoch: {epoch}")
                print(f"Error: {epoch_error}; Threshold: {threshold}")
                print(f"Error: {epoch_error < threshold}")

            if epoch_error < threshold:
                break

    def forward_online(self, x):

        # layer 1 : FUZZIFICATION
        self.layer_1A = self.layer_1_forward_online(x)

        # layer 2 : AGGREGATION
        self.layer_2A = self.layer_2_forward_online()

        # layer 3 : NORMALIZATION
        self.layer_3 = self.layer_3_forward_online()

        # layer 4 : TSK
        self.layer_4A = self.layer_4_forward_online(x)

        self.layer_5Z = self.layer_4A * self.weights_4to5

        # layer 5 : Summ
        self.layer_5A = activate(self.l5_activation_func, np.sum(self.layer_5Z, axis=0))

        res = self.layer_5A
        return res

    def backprop_online(self, learning_rate=0.01):


        dE_by_A5 = self.dE_by_A5
        dA5_by_Z5 = derivative(self.l5_activation_func, self.layer_5A)
        dZ5_by_W4to5 = self.layer_4A  # temporary no bias neuron

        grads_l5 = dE_by_A5 * dA5_by_Z5  # gradients by output neurons

        # 1. Prepare deltas for weights_4to5

        dE_by_W4to5 = np.array(  # deltas for updating weights
            [
                [grads_l5[i] * dZ5_by_W4to5[j] for j in range(dZ5_by_W4to5.shape[0])]
                for i in range(grads_l5.shape[0])
                ]
        ).T[0]

        # 2. Prepare updates for L4 TSK params

        dA4_by_Z4 = derivative("linear", self.layer_4A)
        dZ4_by_L4Params = self.layer_3to4_A

        dE_by_Z5 = grads_l5
        dZ5_by_A4 = self.weights_4to5

        dE_by_A4 = np.array(np.dot(np.matrix(dZ5_by_A4), dE_by_Z5))[0]

        grads_l4 = dE_by_A4 * dA4_by_Z4

        dE_by_L4Params = np.array([grads_l4[i] * dZ4_by_L4Params[i] for i in range(len(grads_l4))])

        # 3. ====================== Prepare error gradients on Layer 2 ===============================================

        dA2_by_Z2 = derivative("linear", self.layer_2A)

        # get partial derivs by var mfs

        _l2_parts_for_derivative_alloc2 = []
        for _rule in range(self.rules.shape[0]):
            _derivs = []
            _a = []
            for _var_num in range(self.rules.shape[1]):
                _d = self.rules[_rule][_var_num]
                _b = self.layer_1A[_var_num][_d]  # get result of fuzzy term * var input
                _a.append(_b)
            for _var_num in range(self.rules.shape[1]):
                _c = copy.copy(_a)
                _c.pop(_var_num)
                _derivs.append(_c)
            _l2_parts_for_derivative_alloc2.append(_derivs)

        _l2_partial_derivs_by_mfs_alloc3 = []
        for i in _l2_parts_for_derivative_alloc2:
            _r = [np.product(derv) for derv in i]
            _l2_partial_derivs_by_mfs_alloc3.append(_r)

        dZ2_by_L1MFs = np.array(_l2_partial_derivs_by_mfs_alloc3)

        dE_by_Z4 = grads_l4
        dZ4_by_A2 = self.layer_2Z

        _l2Z = np.array(self.layer_2Z)

        dE_by_A2 = np.array(
            np.ones(_l2Z.shape) * np.array(np.matrix(dE_by_Z4).T)
        )

        # grads_l2 = dE_by_A2 * dA2_by_Z2
        grads_l2 = dE_by_A2  # dA2_by_Z2 contains only zeroes, ignore it

        # 3. Prepare updates for L1 MF params

        dA1_by_Z1 = np.array([derivative("gaussian", i) for i in self.layer_1A])
        dZ1_by_W_XtoMF = self.layer_0A

        dE_by_Z2 = grads_l2
        dZ2_by_A1 = dZ2_by_L1MFs

        dE_by_A1 = np.array(np.dot(np.matrix(dZ2_by_A1), np.matrix(dE_by_Z2)))[0]

        grads = dE_by_A1 * dA1_by_Z1

        dE_by_W_XtoMF = np.array(
            [[grads[i] * dZ1_by_W_XtoMF[j] for j in range(len(dZ1_by_W_XtoMF))] for i in range(len(grads))])

        # Tune params

        self.weights_Xto1 -= learning_rate * dE_by_W_XtoMF
        self.layer_4_params -= learning_rate * dE_by_L4Params
        self.weights_4to5 -= learning_rate * dE_by_W4to5

        pass

    def layer_1_forward_online(self, x):
        # result1 = np.array(self.memClass.evaluate_mf(x))
        rr = []
        for i in range(len(x)):
            aa = np.array(self.weights_Xto1[i])
            aa[:, 0] = aa[:, 0] * x[i]
            aa = np.sum(aa, axis=1)
            rr.append(aa)

        self.layer_0A = np.append(x, 1)

        return [activate("gaussian", i) for i in rr]

    def layer_2_forward_online(self):
        self.layer_2Z = [
            [
                self.layer_1A[var_num][self.rules[fuzzy_term][var_num]]  # get result of fuzzy term * var input
                for var_num in range(self.rules.shape[1])  # for each column in rule. rules.shape[1] = vars len
                ]
            for fuzzy_term in range(self.rules.shape[0])  # rules rows.shape[0] = mfs terms count
            ]

        # ALL Rules w MFs combinations
        return np.array([np.product(r) for r in self.layer_2Z]).T  # weights

    def layer_3_forward_online(self):
        weights_l2_sum = np.sum(self.layer_2A)  # float
        return self.layer_2A / weights_l2_sum

    def layer_4_forward_online(self, x):
        x__with__bias = np.append(x, 1)
        wn_x = np.array([wght_n * x__with__bias for wght_n in self.layer_3])

        if self.layer_4_params is None:
            self.layer_4_params = np.random.normal(0, 0.3, wn_x.shape)

        if self.weights_4to5 is None:
            self.weights_4to5 = np.ones((wn_x.shape[0], 1))

        self.layer_3to4_A = wn_x

        self.layer_4Z = wn_x * self.layer_4_params

        return np.array(np.matrix(np.sum(self.layer_4Z, axis=1)).T)

    def predict_online(self, Xs, Ys=None):
        errors = []
        predicted = [self.forward_online(Xs[i]) for i in range(Xs.shape[0])]

        if Ys is not None:
            errors = Ys - predicted

        return predicted, errors
