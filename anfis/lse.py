import numpy as np


def LSE(A, B, initialGamma=1000.):
    coefficients_matrix = A  # layer 4
    rhs_matrix = B  # right hand matrix. ie Ys

    S = np.eye(coefficients_matrix.shape[1]) * initialGamma
    # S1 = np.eye(coefficients_matrix.shape[1]) * initialGamma
    x = np.zeros((coefficients_matrix.shape[1], 1))  # need to correct for multi-dim B
    # x1 = np.zeros((coefficients_matrix.shape[1], 1))  # need to correct for multi-dim B

    for i in range(coefficients_matrix.shape[0]):
        a = coefficients_matrix[i, :]  # row of wn * extended_x for each sample_set.
        b = np.array(rhs_matrix[i])

        # TEMP

        # am = np.matrix(a)  # convert ndarray a to matrix a
        # amT = np.matrix(a).T  # transpose a row into column
        # S_x_amT = np.dot(S1, amT)  # dot a with
        #
        # S_x_amT__x__am = np.dot(S_x_amT, am)
        # S_x_amT__x__am___x___S = np.dot(S_x_amT__x__am, S1)
        #
        # S_x_a = np.dot(S1, a)
        # S_x_a__x__a = np.dot(S_x_a, a)
        #
        # _1___pl___S_x_a__x__a = 1 + S_x_a__x__a
        #
        # S1 -= (np.array(S_x_amT__x__am___x___S)) / _1___pl___S_x_a__x__a

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

        # a_x_X = np.dot(np.matrix(a), x)
        # b___minus___a_x_X = np.matrix(b) - a_x_X
        # aT___dot___b___minus___a_x_X = np.dot(np.matrix(a).transpose(), b___minus___a_x_X)
        #
        # x1 += np.dot(S, aT___dot___b___minus___a_x_X)

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
