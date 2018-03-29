import itertools
import numpy as np
import copy
import matplotlib.pyplot as plt

from membership import mf_derivs


def LSE(A, B, initialGamma=1000.):
    coefficients_matrix = A
    rhs_matrix = B

    S = np.eye(coefficients_matrix.shape[1]) * initialGamma
    S1 = np.eye(coefficients_matrix.shape[1]) * initialGamma
    x = np.zeros((coefficients_matrix.shape[1], 1))  # need to correct for multi-dim B
    x1 = np.zeros((coefficients_matrix.shape[1], 1))  # need to correct for multi-dim B

    for i in range(coefficients_matrix.shape[0]):
        a = coefficients_matrix[i, :]  # row of wn * xs for each sample_set
        b = np.array(rhs_matrix[i])

        # TEMP

        # a_transposed = np.matrix(a).transpose()
        # S_x_aT = np.dot(S1, a_transposed)
        # S_x_aT___x___a = np.dot(S_x_aT, np.matrix(a))
        # S_x_aT___x___a__x__S = np.dot(S_x_aT___x___a, S1)
        #
        # S_x_a = np.dot(S1, a)
        # S_x_a__x__a = np.dot(S_x_a, a)
        #
        # _1___pl___S_x_a__x__a = 1 + S_x_a__x__a
        #
        # S1 -= (np.array(S_x_aT___x___a__x__S)) / _1___pl___S_x_a__x__a

        # TEMP end

        S -= (
            (np.array(
                np.dot(
                    np.dot(
                        np.dot(
                            S, np.matrix(a).transpose()
                        ), np.matrix(a)
                    ), S))) /

            (1 + (np.dot(np.dot(S, a), a)))
        )

        # TEMP

        a_x_X = np.dot(np.matrix(a), x)
        b___minus___a_x_X = np.matrix(b) - a_x_X
        aT___dot___b___minus___a_x_X = np.dot(np.matrix(a).transpose(), b___minus___a_x_X)

        x1 += np.dot(S, aT___dot___b___minus___a_x_X)

        # TEMP end

        x += np.dot(
            S, np.dot(
                np.matrix(a).transpose(),
                (np.matrix(b) - np.dot(
                    np.matrix(a), x)
                 )
            )
        )

        # print(S1 == S)

    return x


class ANFIS:
    """Class to implement an Adaptive Network Fuzzy Inference System: ANFIS"

    Attributes:
        X
        Y
        XLen
        memClass
        memFuncs
        memFuncsByVariable
        rules
        consequents
        errors
        memFuncsHomo
        trainingType


    """

    def __init__(self, X, Y, memFunction):
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
            plt.clf()
            plt.plot(range(len(self.errors)), self.errors, 'ro', label='errors')
            plt.ylabel('error')
            plt.xlabel('epoch')
            plt.show()

    def plotMF(self, x, input_var, func):
        plt.clf()
        for mf in range(len(self.memFuncs[input_var])):
            y = func(x, **self.memClass.MFList[input_var][mf])
            plt.plot(x, y, 'r')
        plt.show()

    def plotResults(self):
        if self.trainingType == 'Not trained yet':
            print(self.trainingType)
        else:
            plt.clf()
            plt.plot(range(len(self.fitted_values)), self.fitted_values, 'r', label='trained')
            plt.plot(range(len(self.Y)), self.Y, 'b', label='original')
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

        layer_4 = np.array(np.array_split(layer_4, count_of_X_samples))  # layer 4 becomes list of list, by count of sample_sets

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
                            _a = (self.X.shape[1] + 1)
                            _b = consequent
                            _c = colY
                            _x_sample = self.X[rowX, :]
                            _x_sample_ext = np.append(_x_sample, 1.)

                            _cons_i1 = (_a * _b)
                            _cons_i2 = ((_a * _b) + _a)

                            _f = self.consequents[_cons_i1:_cons_i2, _c]

                            fConsequent = np.dot(_x_sample_ext, _f)
                            _fcon = fConsequent

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
        mi_alloc = [
            [
                layer_1[var_num][self.rules[fuzzy_term][var_num]]  # get result of fuzzy term * var input
                for var_num in range(self.rules.shape[1])  # for each column in rule. rules.shape[1] = vars len
                ]
            for fuzzy_term in range(self.rules.shape[0])  # rules rows.shape[0] = mfs terms count
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
        [layerFour, wSum, w] = self.forward_half_pass(varsToTest)

        # layer five
        layerFive = np.dot(layerFour, self.consequents)

        return layerFive

    def predict_no_learn(self, varsToTest, Y):
        layer_4, weights_l2_sums, weights_l2 = self.forward_half_pass(varsToTest)

        layer_5 = np.dot(layer_4, self.consequents)

        error = np.sqrt(np.sum((Y - layer_5.T) ** 2) / Y.shape[0])
        print('prediction error: ', error)

        return layer_5
