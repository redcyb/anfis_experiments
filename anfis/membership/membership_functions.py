# -*- coding: utf-8 -*-
"""
@author: Alex Marusyk
based on: tim.meggs
"""

import skfuzzy as functions
import numpy as np


class MemberFunction:
    def __init__(self, func, params):
        self.func = getattr(functions, func)
        self.params = params

    def __str__(self):
        return f"{self.func.__name__}: {self.params}"

    def __repr__(self):
        return self.__str__()

    def __call__(self, *args, **kwargs):
        return self.evaluate_mf_for_var(args[0])

    def evaluate_mf_for_var(self, var):
        return self.func(var, *self.params)

    def partial_dmf(self, x, partial_parameter):
        """Calculates the partial derivative of a membership function at a point x."""

        if self.func == functions.gaussmf:

            sigma = self.params[0]
            mean = self.params[1]

            if partial_parameter == 0:
                result = (2. / sigma ** 3) * np.exp(-(((x - mean) ** 2) / sigma ** 2)) * (x - mean) ** 2
            elif partial_parameter == 1:
                result = (2. / sigma ** 2) * np.exp(-(((x - mean) ** 2) / sigma ** 2)) * (x - mean)
            else:
                raise Exception(f"Unknown parameter {partial_parameter}")

        elif self.func == functions.gbellmf:

            a = self.params[0]
            b = self.params[1]
            c = self.params[2]

            if partial_parameter == 0:
                result = (
                    (2. * b * np.power((c - x), 2) * np.power(np.absolute((c - x) / a), ((2 * b) - 2))) /
                    (np.power(a, 3) * np.power((np.power(np.absolute((c - x) / a), (2 * b)) + 1), 2))
                )
            elif partial_parameter == 1:
                result = (
                    -1 * (2 * np.power(np.absolute((c - x) / a), (2 * b)) * np.log(np.absolute((c - x) / a))) /
                    (np.power((np.power(np.absolute((c - x) / a), (2 * b)) + 1), 2))
                )
            elif partial_parameter == 2:
                result = (
                    (2. * b * (c - x) * np.power(np.absolute((c - x) / a), ((2 * b) - 2))) /
                    (np.power(a, 2) * np.power((np.power(np.absolute((c - x) / a), (2 * b)) + 1), 2))
                )
            else:
                raise Exception(f"Unknown parameter {partial_parameter}")

        elif self.func == functions.sigmf:

            b = self.params[0]
            c = self.params[1]

            if partial_parameter == 0:
                result = -1 * (c * np.exp(c * (b + x))) / np.power((np.exp(b * c) + np.exp(c * x)), 2)
            elif partial_parameter == 1:
                result = ((x - b) * np.exp(c * (x - b))) / np.power((np.exp(c * (x - c))) + 1, 2)
            else:
                raise Exception(f"Unknown parameter {partial_parameter}")

        else:
            raise Exception("Unknown function")

        return result


class MemFuncs:
    func = None

    def __init__(self, func, mfs_list):
        self.mfs_list = mfs_list
        self.func = func
        print(func.__name__)

    def evaluate_mf(self, sample_set):
        if not self.func:
            raise NotImplementedError("There is no function to call")

        if len(sample_set) != len(self.mfs_list):
            print("Number of variables does not match number of rule sets")

        return [
            [
                self.func(sample_set[i], **self.mfs_list[i][k])
                for k in range(len(self.mfs_list[i]))  # apply K FuzzyTerm part of MF
                ]
            for i in range(len(sample_set))  # to I input var in sample set
            ]


class Layer1:
    inputs = []

    def __init__(self, inputs):
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def evaluate_mf(self, sample_set):
        if len(sample_set) != len(self.inputs):
            print("Number of variables does not match number of rule sets")

        return [
            [
                self.inputs[inp_num]["mfs"][mf_num](sample_set[inp_num])
                for mf_num in range(len(self.inputs[inp_num]["mfs"]))  # apply K FuzzyTerm part of MF
                ]
            for inp_num in range(len(sample_set))  # to I input var in sample set
            ]


def evaluateMFforVar(func, MFListForVar, var):
    return [
        func(var, **MFListForVar[k])  # apply K part of MF to only one var
        for k in range(len(MFListForVar))
        ]
