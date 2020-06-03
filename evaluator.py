# -*- coding: utf-8 -*-
"""
@Description :
@Time : 2020/4/12 12:35 
@Author : Kunsong Zhao
@Version：1.0
"""

__author__ = 'kszhao'

from measures import *


class Evaluator:

    def __init__(self, y_pred, y, measure_names):
        self.y_pred = y_pred
        self.y = y
        self.measure_names = measure_names

    @staticmethod
    def evaluate(measure_names, y_pred, y, result, project, algo, count):
        for measure in measure_names:
            result[project][count][algo][measure] = \
                str(eval(measure)(y, y_pred))

        return result


