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
    weights_XtoMFs = None
    dE_by_A5 = None

    l5_activation_func = None

    predicted = None

    def __init__(self, X, Y, memFunction=None, l5_activation_func="linear"):
        self.X = X
        self.Y = Y

        self.errors = np.empty(0)
        self.trainingType = 'Not trained yet'

        # TEMP!!!!!
        self.memClass = copy.deepcopy(memFunction)

        # NEW

        self.l5_activation_func = l5_activation_func
        input_vars_ranges = [[X[:, i].min(), X[:, i].max()] for i in range(X.shape[1])]

        num_of_mfs_for_each_var = 3

        self.weights_XtoMFs = [np.array(gen_mf_by_range("gaussian", num_of_mfs_for_each_var, *i)) for i in
                               input_vars_ranges]
        self.mfs_by_vars_combinations = [[x for x in range(num_of_mfs_for_each_var)] for z in range(X.shape[1])]
        self.rules = np.array(list(itertools.product(*self.mfs_by_vars_combinations)))

        pass

    def predict_online(self, Xs, Ys=None):
        errors = []
        predicted = [self.forward_online(Xs[i]) for i in range(Xs.shape[0])]

        if Ys is not None:
            errors = Ys - predicted

        return predicted, errors

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

    def layer_1_forward_online(self, x):
        # result1 = np.array(self.memClass.evaluate_mf(x))
        rr = []
        for i in range(len(x)):
            aa = np.array(self.weights_XtoMFs[i])
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

        self.weights_XtoMFs -= learning_rate * dE_by_W_XtoMF
        self.layer_4_params -= learning_rate * dE_by_L4Params
        self.weights_4to5 -= learning_rate * dE_by_W4to5

        pass

    def layer_5_backprop(self):
        pass

    def layer_4_backprop(self):
        pass

    def layer_3_backprop(self):
        pass

    def layer_2_backprop(self):
        pass

    def layer_1_backprop(self):
        pass
