__author__ = "Alex Marusyk"
__copyright__ = "Copyright 2018, Alex Marusyk"
__license__ = "MIT"
__version__ = "0.0.1"
__email__ = "redcyb@gmail.com"
__status__ = "development"

import numpy as np

from functions import activate, derivative


class MLPerceptron:
    def __init__(self, Xs, Ys,
                 layers_config={"hidden": ((2, "sigmoid"),), "output": "sigmoid"},
                 weights_dispersion=0.2,
                 **kwargs):

        """
        :param Xs: np.array
        :param Ys: np.array
        :param layers_config: config dict with settings of layers from second to outputs
        :param weights_dispersion: float in [0, 1] for weights initialization
        :param kwargs: dict

        Properties:

        layers_A = Activated Output of each layer: layers_A[l][i] = activation_func(layers_Z[l][i])
        layers_Z = Weighted Sum of inputs of each layer: sum(xi * wji)

        self.AOs = Array of predicted Outputs for whole dataset

        weights = weights between all Layers
        weights_deltas = weights deltas to update weights after each dataset example handling

        self.EO = 1/2 * SUM(Ysn - self.AO)^2 for 1 Dataset Example (online). I.e. errors by output neurons
        self.AO_m_Y = dEO / dAO   where AO means layers_A[-1] aka last layer or output layer

        self.ETotal = sum(self.EO) for 1 Dataset Example (online).

        """

        self.layers_config = layers_config

        inputs_size = Xs.shape[1]
        outputs_size = Ys.shape[1]
        hidden_shape = [i[0] for i in layers_config["hidden"]]

        self.layers_A = np.array(
            [[None] * (inputs_size + 1), *[[None] * (i + 1) for i in hidden_shape], [None] * outputs_size]
        )

        self.layers_Z = np.array(
            [[None] * inputs_size, *[[None] * i for i in hidden_shape], [None] * outputs_size]
        )

        self.weights = [np.random.normal(0, weights_dispersion, (
            len(self.layers_A[i]) - 1 if i < self.layers_A.shape[0] - 1 else len(self.layers_A[i]),
            len(self.layers_A[i - 1])
        )) for i in range(1, self.layers_A.shape[0])]

        self.weights_deltas = [np.ones((
            len(self.layers_A[i]) - 1 if i < self.layers_A.shape[0] - 1 else len(self.layers_A[i]),
            len(self.layers_A[i - 1])
        )) for i in range(1, self.layers_A.shape[0])]

        self.AO_m_Y = []
        self.EO = []
        self.ETotal = []

        self.AOs = []

    def forward(self, Xsn, Ysn):
        self.layers_A[0] = np.append(Xsn, 1)

        for i in range(1, self.layers_A.shape[0]):
            a = self.weights[i - 1]
            b = self.layers_A[i - 1]
            c = np.dot(a, b)

            self.layers_Z[i] = c

            if i < self.layers_A.shape[0] - 1:
                act = np.append(activate(self.layers_config["hidden"][i - 1][1], self.layers_Z[i]), 1)
            else:
                act = activate(self.layers_config["output"], self.layers_Z[i])

            self.layers_A[i] = act
            pass

        self.AO_m_Y = self.layers_A[-1] - Ysn
        self.EO = np.array(
            np.power((Ysn - self.layers_A[-1]), np.array([2 for i in range(len(self.layers_A[-1]))])) / 2)
        self.ETotal = np.sum(self.EO)

        pass

    def backward(self, step=0.5):
        # Layer H-O

        dE_by_dO = self.AO_m_Y  # dEi/dOut for squared sum loss function
        dAO_by_ZO = derivative(self.layers_config["output"], self.layers_Z[-1])
        dZO_by_HOW = self.layers_A[-2]
        grads = dE_by_dO * dAO_by_ZO
        dE_by_HOW = np.array([[grads[i] * dZO_by_HOW[j] for j in range(len(dZO_by_HOW))] for i in range(len(grads))])
        self.weights_deltas[-1] = step * dE_by_HOW

        for k in range(len(self.layers_A) - 2, 0, -1):
            dAk_by_Zk = derivative(self.layers_config["hidden"][k - 1][1], self.layers_Z[k])
            dZk_by_Wk = self.layers_A[k - 1]  # just activated outputs of previous layer

            dEn_by_Zn = grads
            dZn_by_Ak = self.weights[k][:, :-1]  # we don't need bias in HO layer for AH

            dE_by_Ak = np.array(np.dot(np.matrix(dZn_by_Ak).T, dEn_by_Zn))[0]

            grads = dE_by_Ak * dAk_by_Zk

            dE_by_Wk = np.array([[grads[i] * dZk_by_Wk[j] for j in range(len(dZk_by_Wk))] for i in range(len(grads))])
            self.weights_deltas[k - 1] = step * dE_by_Wk

        # update all weights

        for i in range(len(self.weights)):
            self.weights[i] -= self.weights_deltas[i]

    def train_online(self, Xs, Ys, epochs=1000, learning_rate=0.1, threshold=0.01, report_every=100, min_epochs=1):
        epoch = 0

        for i in range(epochs):
            epoch += 1

            for n in range(Xs.shape[0]):
                self.forward(Xs[n], Ys[n])
                self.backward(learning_rate)

            if self.ETotal < threshold and epoch > min_epochs:
                break

            if report_every and not epoch % report_every:
                print(f"\nEpoch: {epoch}")
                print(f"Error: {self.ETotal}")

        print(f"\nTraining epochs: {epoch}")
        print(f"Training Error: {self.ETotal}\n")

    def predict(self, Xs, Ys):
        print("Predicted:\n")

        for n in range(Xs.shape[0]):
            self.forward(Xs[n], Ys[n])
            self.AOs.append(self.layers_A[-1])
            print(f"X: {Xs[n]}    O: {[f'{i:.2f}' for i in self.layers_A[-1]]}    Y: {[f'{i:.2f}' for i in Ys[n]]}")

        self.AOs = np.array(self.AOs)

        print(f"\nTest Error: {self.ETotal}\n")
