import numpy as np


def LSE(A, B, initialGamma=1000.):
    coefficients_matrix = A  # layer 4
    rhs_matrix = B  # right hand matrix. ie Ys

    S = np.eye(coefficients_matrix.shape[1]) * initialGamma
    x = np.zeros((coefficients_matrix.shape[1], 1))  # need to correct for multi-dim B

    for i in range(coefficients_matrix.shape[0]):
        a = coefficients_matrix[i, :]  # row of wn * extended_x for each sample_set.
        am = np.matrix(a)

        b = np.array(rhs_matrix[i])
        bm = np.matrix(b)

        S -= (np.dot(np.dot(np.dot(S, am.T), am), S)) / (1 + np.dot(np.dot(S, a), a))
        x += np.dot(S, np.dot(am.T, (bm - np.dot(am, x))))

        # print()

    return x
