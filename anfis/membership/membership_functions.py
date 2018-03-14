# -*- coding: utf-8 -*-
"""
@author: Alex Marusyk
based on: tim.meggs
"""

import skfuzzy as functions


class MemberFunction:
    def __init__(self, func, params):
        self.func = getattr(functions, func)
        self.params = params

    def evaluate_mf_for_var(self, var):
        return self.func(var, *self.params)

    def __str__(self):
        return f"{self.func.__name__} :: {self.params}"

    def __repr__(self):
        return self.__str__()


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


def evaluateMFforVar(func, MFListForVar, var):
    return [
        func(var, **MFListForVar[k])                       # apply K part of MF to only one var
        for k in range(len(MFListForVar))
    ]