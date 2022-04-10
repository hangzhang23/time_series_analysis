# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/04/11 02:34
# @Author  : TJD
# @FileName: base_class.py
import abc


class ForecastingBase:
    """
    钢铁消费量预测基类
    """
    @abc.abstractmethod
    def data_analysis(self):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def validate(self):
        pass

    @abc.abstractmethod
    def predict(self):
        pass


class ModelBase:
    """
    模型训练基类
    """
    @abc.abstractmethod
    def _preprocessing(self):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate(self, model):
        pass