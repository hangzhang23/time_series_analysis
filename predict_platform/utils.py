# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/04/16 02:26
# @Author  : TJD
# @FileName: utils.py
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def create_dirs(save_path):
    try:
        os.makedirs(save_path)
    except FileExistsError:
        os.path.exists(save_path)


def regression_evaluate(gt, pred, method='mse'):
    """
    评估回归结果
    :param gt:
    :param pred:
    :return:
    """
    if method is 'mse':
        score = mean_squared_error(gt, pred, squared=False)
    elif method is 'mae':
        score = mean_absolute_error(gt, pred)
    return score


def up_down_acc(gt_train, pred_train, gt_test, pred_test):
    """
    评估预测涨跌准确性
    :param gt:
    :param pred:
    :return:
    """
    # 训练集
    df = pd.DataFrame(columns=['gt', 'pred'])
    df['gt'] = np.concatenate([gt_train, gt_test])
    df['pred'] = np.concatenate([pred_train, pred_test])
    df['gt_diff'] = df['gt'] - df['gt'].shift(1)
    df['pred_diff'] = df['pred'] - df['pred'].shift(1)
    df['diff_judge'] = (df['gt_diff'] * df['pred_diff']).apply(lambda x: 1 if x > 0 else 0)
    df['acc_train'] = df['diff_judge'][:len(gt_train)].mean()
    df['acc_test'] = df['diff_judge'][-len(gt_test):].mean()
    return df


def reciprocity_verify(consump_t_file, consump_v_file, x_file, y_file, date_flag, result_flag):
    '''
    自动对训练和验证的需求量进行对应性检验
    :param consump_t_file:训练集的需求量
    :param consump_v_file:验证集的消费量
    :param x_file:输入给nn的数据集x（最后一列是产量）
    :param y_file:对应的价格涨跌量y
    :return: None
    '''
    consumpt_trian = pd.read_excel(consump_t_file, sheet_name='Sheet1')
    consumpt_val = pd.read_excel(consump_v_file, sheet_name='Sheet1')
    production = pd.read_excel(x_file, sheet_name='Sheet1')['产量:钢材:当月值']
    price = pd.read_excel(y_file, sheet_name='Sheet1')
    # 拼接数据
    temp = pd.concat([pd.concat([consumpt_trian, consumpt_val]).reset_index(drop=True), production], axis=1).iloc[:, :]
    df = pd.concat([price, temp], axis=1)
    # 验证对应性
    df['需求产量差'] = (df['consumption'] - df['产量:钢材:当月值']).apply(lambda x: 1 if x >= 0 else 0)
    df['验证情况'] = (df['需求产量差'] - df['price']).apply(lambda x: 1 if x == 0 else 0)
    df['训练集对应准确率'] = df['验证情况'][:consumpt_trian.shape[0]].mean()
    df['验证集对应准确率'] = df['验证情况'][-consumpt_val.shape[0]:].mean()
    # 输出结果
    df.to_excel('./temp/' + date_flag + '/consumption_t&v_verify_' + date_flag + '_'
                  + result_flag + '.xlsx', header=True)
    print('训练集（长度{}）对应准确率:{}'.format(consumpt_trian.shape[0], df['训练集对应准确率'][0]))
    print('验证集（长度{}）对应准确率:{}'.format(consumpt_val.shape[0], df['验证集对应准确率'][0]))
    return df