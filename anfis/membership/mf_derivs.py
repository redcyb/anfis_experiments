import numpy as np


def partial_dMF(x, mf_definition, partial_parameter):
    """Calculates the partial derivative of a membership function at a point x.

    Parameters
    ------

    Returns
    ------

    """

    if mf_name == 'gaussmf':

        sigma = mf_definition['sigma']
        mean = mf_definition['mean']

        if partial_parameter == 'sigma':
            result = (2. / sigma ** 3) * np.exp(-(((x - mean) ** 2) / (sigma) ** 2)) * (x - mean) ** 2
        elif partial_parameter == 'mean':
            result = (2. / sigma ** 2) * np.exp(-(((x - mean) ** 2) / (sigma) ** 2)) * (x - mean)
        else:
            raise Exception(f"Unknown parameter {partial_parameter}")

    elif mf_name == 'gbellmf':

        a = mf_definition['a']
        b = mf_definition['b']
        c = mf_definition['c']

        if partial_parameter == 'a':
            result = (2. * b * np.power((c - x), 2) * np.power(np.absolute((c - x) / a), ((2 * b) - 2))) / \
                     (np.power(a, 3) * np.power((np.power(np.absolute((c - x) / a), (2 * b)) + 1), 2))
        elif partial_parameter == 'b':
            result = -1 * (2 * np.power(np.absolute((c - x) / a), (2 * b)) * np.log(np.absolute((c - x) / a))) / \
                     (np.power((np.power(np.absolute((c - x) / a), (2 * b)) + 1), 2))
        elif partial_parameter == 'c':
            result = (2. * b * (c - x) * np.power(np.absolute((c - x) / a), ((2 * b) - 2))) / \
                     (np.power(a, 2) * np.power((np.power(np.absolute((c - x) / a), (2 * b)) + 1), 2))
        else:
            raise Exception(f"Unknown parameter {partial_parameter}")

    elif mf_name == 'sigmf':

        b = mf_definition['b']
        c = mf_definition['c']

        if partial_parameter == 'b':
            result = -1 * (c * np.exp(c * (b + x))) / np.power((np.exp(b * c) + np.exp(c * x)), 2)
        elif partial_parameter == 'c':
            result = ((x - b) * np.exp(c * (x - b))) / np.power((np.exp(c * (x - c))) + 1, 2)
        else:
            raise Exception(f"Unknown parameter {partial_parameter}")

    else:
        raise Exception("Unknown function")

    return result
