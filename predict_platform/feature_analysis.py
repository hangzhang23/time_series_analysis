# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/04/11 06:17
# @Author  : TJD
# @FileName: feature_analysis.py
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
import numpy as np
import math
from hyperopt import STATUS_OK, Trials, fmin, tpe
from matplotlib import pyplot as plt
from utils import *
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class TreeAnalysis:
    """
    通用树模型判断特征重要性工具，可根据输入y自动判断使用分类树还是回归树，并且加入贝叶斯优化来自动调整超参。
    输出结果包括：
    1、一个参数空间的分析图；
    2、一个最终的特征重要性分析图；
    3、相关性图和数据分布箱型图；
    4、调整完参数的训练和预测拟合情况图。
    """
    def __init__(self, x_data, y_data, dt_space):
        self.x_data = x_data
        self.target = y_data
        self.dt_hp_space = dt_space
        self.scenario = self.__target_judge()
        self.feature_names = self.x_data.columns[1:]

    def __target_judge(self):
        """
        根据输入y判断当前树模型的任务是分类还是回归。
        :return:
        """
        count_dic = dict(self.target.iloc[:, 1].value_counts())
        if len(count_dic.keys()) == 2:
            print('根据目标值判断为分类任务')
            return 'clf'
        elif len(count_dic.keys()) > 2:
            print('根据目标值判断为回归任务')
            return 'reg'
        else:
            raise ValueError('目标值统计只能大于等于2，目前目标值统计为{}'.format(len(count_dic.keys())))

    def _data_preprocessing(self, x, y, split_percent):
        """
        由于树模型主要计算的是信息增益，所以特征不需要进行变化。这里主要作用是做分割
        :param x:
        :param y:
        :param split_percent:
        :return:
        """
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_percent, shuffle=False)
        return x_train, x_test, y_train, y_test

    def _tree_training(self, params, x_train, x_test, y_train, y_test):
        """
        根据关键字判断是建立回归树还是分类树，通过训练集和验证集来判断拟合情况。
        在贝叶斯优化中，导入需要调整的参数。在固定参数的情况中，导入固定的参数。
        :param params:
        :param x_train:
        :param x_test:
        :param y_train:
        :param y_test:
        :return:
        """
        if self.scenario is 'clf':
            tree = DecisionTreeClassifier(**params)
            tree.fit(x_train, y_train)
            y_pred = tree.predict(x_test)
            score = accuracy_score(y_test, y_pred)
        elif self.scenario is 'reg':
            tree = DecisionTreeRegressor(**params)
            tree.fit(x_train, y_train)
            y_pred = tree.predict(x_test)
            score = mean_squared_error(y_test, y_pred, squared=False)

        return -score if self.scenario is 'clf' else score, tree

    def _bayes_hpopt(self, params):
        x_train, x_test, y_train, y_test = self._data_preprocessing(self.x_data.iloc[:len(self.target), 1:],
                                                                    self.target.iloc[:, 1],
                                                                    split_percent=0.1)
        score, _ = self._tree_training(params, x_train, x_test, y_train, y_test)
        return {'loss': score, 'status': STATUS_OK}

    def _bayes_hpsearch(self):
        trials = Trials()
        best = fmin(self._bayes_hpopt, self.dt_hp_space, algo=tpe.suggest, max_evals=300,
                    trials=trials)
        self._hpopt_plot(trials)
        return best

    def _hpopt_plot(self, trials):
        parameters = ['max_depth', 'max_features', 'criterion']
        cols = len(parameters)
        f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(20, 5))
        cmap = plt.cm.jet
        for i, val in enumerate(parameters):
            xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
            ys = [-t['result']['loss'] for t in trials.trials]
            axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.25, c=cmap(float(i)/len(parameters)))
            axes[i].set_title(val)
            create_dirs('./tmp/feature')
        plt.savefig('./tmp/feature/hp_analysis.png')
        print('贝叶斯优化参数图生成完毕。')

    def _correlation_heatmap(self, x, y, method='pearson'):
        """
        x和y的相关性图展示
        :param x:
        :param y:
        :param method:
        :return:
        """
        df = pd.concat([x, y.iloc[:, 1]], axis=1)
        plt.figure(figsize=(20, 10))
        plt.title('所有特征和目标值之间的' + method + '相关系数')
        sns.heatmap(df.corr(method), linecolor='white', linewidths=0.1, cmap='RdBu', vmin=-1, vmax=1)
        create_dirs('./tmp/feature')
        plt.savefig('./tmp/feature/{}_correlation_map.png'.format(method))
        print('相关性图生成完毕。')

    def _distribution_plot(self, x, y):
        """
        箱型分布图展示
        :param x:
        :param y:
        :return:
        """
        df = pd.concat([x, y.iloc[:, 1]], axis=1)
        row_count = math.ceil(x.shape[1] / 4)
        f, axes = plt.subplots(row_count, 4, figsize=(24, row_count * 6))
        count = 0
        for i in x.columns.tolist():
            sns.boxenplot(x=i, y=y.columns[-1], data=df, ax=axes[count // 4][count % 4])
            count += 1
        create_dirs('./tmp/feature')
        plt.savefig('./tmp/feature/boxedplot.png')
        print('特征箱型分布图生成完毕。')

    def _feature_importance(self, model):
        """
        模型训练完毕后输出重要性条形图。
        :param model:
        :return:
        """
        importance = model.feature_importances_
        sorted_indices = importance.argsort()
        # plot importance
        y_ticks = np.arange(0, len(self.feature_names))
        fig, ax = plt.subplots(figsize=(30, 30))
        ax.set_title('feature importance')
        ax.barh(y_ticks, importance[sorted_indices])
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(self.feature_names[sorted_indices])
        fig.tight_layout()
        create_dirs('./tmp/feature')
        plt.savefig('./tmp/feature/feature_importance.png')
        print('特征重要性图生成完毕')

    def feature_analysis(self):
        """
        通过贝叶斯优化调参，并输出特征重要性图和结果。
        :return:
        """
        x_train, x_test, y_train, y_test = self._data_preprocessing(self.x_data.iloc[:len(self.target), 1:],
                                                                    self.target.iloc[:, 1],
                                                                    split_percent=0.1)
        best_params = self._bayes_hpsearch()
        if self.scenario is 'clf':
            best_params['criterion'] = 'gini' if best_params['criterion'] == 0 else 'entropy'
        elif self.scenario is 'reg':
            if best_params['criterion'] == 0:
                best_params['criterion'] = 'mse'
            elif best_params['criterion'] == 1:
                best_params['criterion'] = 'friedman_mse'
            elif best_params['criterion'] == 2:
                best_params['criterion'] = 'mae'

        print('best hyperparameters after searching:', best_params)
        # train tree model
        _, model = self._tree_training(best_params, x_train, x_test, y_train, y_test)
        self.fitting_plot(x_train, x_test, y_train, y_test, model)
        self._feature_importance(model)

    def feature_exploring(self):
        """
        探索最基本的x和y之间的关系，更详细的建议用jupyter来做。
        :return:
        """
        # 查看数据分布
        self._distribution_plot(self.x_data.iloc[:len(self.target), 1:], self.target)
        # 查看相关性
        self._correlation_heatmap(self.x_data.iloc[:len(self.target), 1:], self.target, method='pearson')

    def fitting_plot(self, x_train, x_test, y_train, y_test, model):
        """
        最优参数训练测试拟合情况展示
        :param x_train:
        :param x_test:
        :param y_train:
        :param y_test:
        :param model:
        :return:
        """
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        figure, axs = plt.subplots(1, 2, figsize=(16, 5))
        axs[0].plot([i for i in range(len(train_pred))], y_train, label='train_gt')
        axs[0].plot([i for i in range(len(train_pred))], train_pred, label='train_pred')
        axs[0].set_title('训练情况')
        axs[0].legend()
        axs[1].plot([i for i in range(len(test_pred))], y_test, label='test_gt')
        axs[1].plot([i for i in range(len(test_pred))], test_pred, label='test_pred')
        axs[1].set_title('验证情况')
        axs[1].legend()
        create_dirs('./tmp/feature')
        plt.savefig('./tmp/feature/fitting_plot.png')
        print('训练和验证拟合情况图生成完毕')

    def feature_select(self):
        """
        根据重要性做特征选择
        :return:
        """
        pass