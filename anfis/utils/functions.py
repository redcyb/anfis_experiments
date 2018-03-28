import numpy as np


def derivative(func_name, x, **kwargs):
    if func_name == "sigmoid":
        y = activate("sigmoid", x)
        return y * (1 - y)

    if func_name == "linear":
        if isinstance(x, (np.ndarray, np.generic)):
            return np.ones(x.shape[0])
        return 1

    if func_name == "tanh":
        return 1. / np.power(np.cosh(x), 2.)

    if func_name == "gaussian":
        mean = 0
        sigma = 1.
        if kwargs.get("by") == "mean":
            # TODO Derive by mean param
            pass
        if kwargs.get("by") == "sigma":
            # TODO Derive by sigma param
            pass
        return (-2 * (x - mean) * np.exp(-((x - mean) ** 2) / (sigma ** 2))) / (sigma ** 2)

    raise Exception(f"Unknown function: {func_name}")


def activate(func_name, x, **kwargs):
    if func_name == "sigmoid":
        c = 1
        b = 0
        return 1. / (1. + np.exp(-c * (x - b)))

    if func_name == "tanh":
        return np.tanh(x)

    if func_name == "linear":
        a = 1.
        b = 0
        return a * x + b

    if func_name == "gaussian":
        mean = 0
        sigma = 1.
        return np.exp(-((x - mean) ** 2.) / (2. * sigma ** 2.))

    raise Exception(f"Unknown function: {func_name}")
