import numpy as np


def partial_dMF(x, mf_definition, partial_parameter, mf_func):
    """Calculates the partial derivative of a membership function at a point x.

    Parameters
    ------

    Returns
    ------

    """

    mf_name = mf_func.__name__

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
            result = -1 * (c * np.exp(c * (b + x))) / \
                     np.power((np.exp(b * c) + np.exp(c * x)), 2)
        elif partial_parameter == 'c':
            result = ((x - b) * np.exp(c * (x - b))) / \
                     np.power((np.exp(c * (x - c))) + 1, 2)
        else:
            raise Exception(f"Unknown parameter {partial_parameter}")

    else:
        raise Exception("Unknown function")

    return result


def partial_dmf(x, mf_name, mf_parameter_dict, partial_parameter):
    """
    Calculate the *partial derivative* of a specified membership function.

    Parameters
    ----------
    x : float
        input variable.
    mf_name : string
        Membership function name as a string. The following are supported:
        * ``'gaussmf'`` : parameters ``'sigma'`` or ``'mean'``
        * ``'gbellmf'`` : parameters ``'a'``, ``'b'``, or ``'c'``
        * ``'sigmf'`` : parameters ``'b'`` or ``'c'``
    mf_parameter_dict : dict
        A dictionary of ``{param : key-value, ...}`` pairs for a particular
        membership function as defined above.
    partial_parameter : string
        Name of the parameter against which we take the partial derivative.

    Returns
    -------
    d : float
        Partial derivative of the membership function with respect to the
        chosen parameter, at input point ``x``.

    Notes
    -----
    Partial derivatives of fuzzy membership functions are only meaningful for
    continuous functions. Triangular, trapezoidal designs have no partial
    derivatives to calculate. The following
    """

    result = None

    if mf_name == 'gaussmf':

        sigma = mf_parameter_dict['sigma']
        mean = mf_parameter_dict['mean']

        if partial_parameter == 'sigma':
            result = ((2. / sigma**3) *
                      np.exp(-(((x - mean)**2) / (sigma)**2)) * (x - mean)**2)
        elif partial_parameter == 'mean':
            result = ((2. / sigma**2) *
                      np.exp(-(((x - mean)**2) / (sigma)**2)) * (x - mean))

    elif mf_name == 'gbellmf':

        a = mf_parameter_dict['a']
        b = mf_parameter_dict['b']
        c = mf_parameter_dict['c']

        # Partial result for speed and conciseness in derived eqs below
        d = np.abs((c - x) / a)

        if partial_parameter == 'a':
            result = ((2. * b * (c - x)**2.) * d**((2 * b) - 2) /
                      (a**3. * (d**(2. * b) + 1)**2.))

        elif partial_parameter == 'b':
            result = (-1 * (2 * d**(2. * b) * np.log(d)) /
                      ((d**(2. * b) + 1)**2.))

        elif partial_parameter == 'c':
            result = ((2. * b * (x - c) * d**((2. * b) - 2)) /
                      (a**2. * (d**(2. * b) + 1)**2.))

    elif mf_name == 'sigmf':

        b = mf_parameter_dict['b']
        c = mf_parameter_dict['c']

        if partial_parameter == 'b':
            # Partial result for speed and conciseness
            d = np.exp(c * (b + x))
            result = -1 * (c * d) / (np.exp(b * c) + np.exp(c * x))**2.

        elif partial_parameter == 'c':
            # Partial result for speed and conciseness
            d = np.exp(c * (x - b))
            result = ((x - b) * d) / (d + 1)**2.

    return result
