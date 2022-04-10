from base_class import ForecastingBase, ModelBase
from feature_analysis import TreeAnalysis
from utils import *

from hyperopt import hp
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from hyperopt import STATUS_OK, Trials, fmin, tpe
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt


class ApperantSCForecast(ForecastingBase):
    def __init__(self, x_data, y_data, x_sheet, date, manual_params=None, y_sheet='Sheet1'):
        self.x_data = pd.read_excel(x_data, sheet_name=x_sheet)
        self.target = pd.read_excel(y_data, sheet_name=y_sheet)
        self.feature_names = self.x_data.columns[1:]
        self.date_flag = date
        self.tree_list = ['DT', 'RF', 'xgb', 'lgb']
        self.other_list = ['LR', 'svm']
        self.manual_params = manual_params

    def data_analysis(self):
        """
        特征分析，用决策树查看基本特征情况和特征选择
        :return:
        """
        # 贝叶斯优化参数空间
        dt_space = {
            'max_depth': hp.choice('max_depth', range(1, 20)),
            'max_features': hp.choice('max_features', range(1, 5)),
            'criterion': hp.choice('criterion', ['mse', 'friedman_mse', 'mae'])
        }
        print('=======================')
        # initiate analyst
        analyst = TreeAnalysis(x_data=self.x_data,
                               y_data=self.target,
                               dt_space=dt_space)
        # data exploring
        analyst.feature_exploring()
        # feature analysing
        analyst.feature_analysis()
        print('-----------------------')
        print('特征分析完毕, 请查看tmp/feature下分析结果。')
        print('=======================')

    def train(self, mode):
        if mode in self.tree_list:
            tree_predictor = TreePredict(x=self.x_data.iloc[:, 1:],
                                         y=self.target.iloc[:, 1],
                                         feature_names=self.feature_names,
                                         tree_mode=mode,
                                         val_size=0.1,
                                         date_flag=self.date_flag,
                                         manual=self.manual_params)
            # 模型训练
            model = tree_predictor.train()
            # 模型效果评估
            direct_acc = tree_predictor.evaluate(model)
        else:
            if mode is 'mlp':
                mlp_predictor = NeuralNetworkPredict(self.x_data, self.target, self.feature_names)
            else:
                raise ValueError('输入mode：{} 为不支持模型'.format(mode))

        return direct_acc

    def predict(self):
        pass


class TreePredict(ModelBase):
    """
    树模型预测，initate中需要指出使用的树模型为随机森林（RF），xgboost（xgb），LightGBM（lgb）。决策树（DT）
    由于拟合效果比其他高级树模型差故没有添加。对象先对指定树模型进行贝叶斯调参，之后对最佳参数组合训练，并评估训练
    和验证的拟合情况以及涨跌准确率。
    """
    def __init__(self, x, y, feature_names, tree_mode, val_size, date_flag, manual=None):
        self.x_data = x
        self.y_data = y
        self.feature_names = feature_names
        self.tree_mode = tree_mode
        self.val_size = val_size
        self.date_flag = date_flag
        self._preprocessing()
        self.manual_params = manual

    def _preprocessing(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_data,
                                                                                self.y_data,
                                                                                test_size=self.val_size,
                                                                                shuffle=False)

    def _tree_training(self, params):
        if self.tree_mode is 'RF':
            tree = RandomForestRegressor(**params)
        elif self.tree_mode is 'xgb':
            tree = XGBRegressor(importance_type='gain',
                                **params)
        elif self.tree_mode is 'lgb':
            tree = LGBMRegressor(importance_type='split',
                                 **params)
        tree.fit(self.x_train, self.y_train)
        y_pred = tree.predict(self.x_test)
        score = mean_squared_error(self.y_test, y_pred, squared=False)
        return score, tree

    def _bayes_hpopt(self, params):
        score, _ = self._tree_training(params)
        return {'loss': score, 'status': STATUS_OK}

    def _bayes_hpsearch(self, dt_space):
        trials = Trials()
        best = fmin(self._bayes_hpopt, dt_space, algo=tpe.suggest, max_evals=120, trials=trials)
        return best

    def _hyperparam_search(self):
        if self.tree_mode is 'RF':
            dt_space = {
                'n_estimators': hp.choice('n_estimators', range(5, 50)),
                'max_depth': hp.choice('max_depth', range(3, 10)),
                'max_features': hp.choice('max_features', range(2, 10)),
                'criterion': hp.choice('criterion', ['mse', 'friedman_mse', 'mae']),
                'min_samples_split': hp.uniform('min_samples_split', 1e-1, 6e-1),
                'min_samples_leaf': hp.uniform('min_samples_leaf', 1e-1, 6e-1)
            }
        elif self.tree_mode is 'xgb':
            dt_space = {
                "max_depth": hp.randint("max_depth", 10),
                "subsample": hp.uniform("subsample", 1e-1, 1),
                "learning_rate": hp.uniform("learning_rate", 1e-1, 7e-1),
                "colsample_bytree": hp.uniform("colsample_bytree", 1e-1, 9e-1),
                "n_estimators": hp.choice("n_estimators", range(10, 100)),
                "min_child_weight": hp.randint("min_child_weight", 100),
            }
        elif self.tree_mode is 'lgb':
            dt_space = {
                "max_depth": hp.choice("max_depth", range(2, 6)),
                "num_leaves": hp.choice("num_leaves", range(3, 20)),
                "subsample": hp.uniform("subsample", 1e-1, 1),
                "learning_rate": hp.uniform("learning_rate", 1e-1, 7e-1),
                "colsample_bytree": hp.uniform("colsample_bytree", 1e-1, 9e-1),
                "n_estimators": hp.choice("n_estimators", range(5, 100)),
                "min_child_weight": hp.randint("min_child_weight", 10)
            }

        best_hp = self._bayes_hpsearch(dt_space)
        return best_hp

    def _fitting_plot(self, train_pred, test_pred):
        """
        最优参数训练测试拟合情况展示
        :param x_train:
        :param x_test:
        :param y_train:
        :param y_test:
        :param model:
        :return:
        """
        figure, axs = plt.subplots(1, 2, figsize=(16, 5))
        axs[0].plot([i for i in range(len(train_pred))], self.y_train, label='train_gt')
        axs[0].plot([i for i in range(len(train_pred))], train_pred, label='train_pred')
        axs[0].set_title('训练情况')
        axs[0].legend()
        axs[1].plot([i for i in range(len(test_pred))], self.y_test, label='test_gt')
        axs[1].plot([i for i in range(len(test_pred))], test_pred, label='test_pred')
        axs[1].set_title('验证情况')
        axs[1].legend()
        create_dirs('./tmp/result/{}'.format(self.date_flag))
        plt.savefig('./tmp/result/{}/fitting_plot_{}.png'.format(self.date_flag, self.tree_mode))
        print('训练和验证拟合情况图生成完毕')

    def _importance_export(self, importance):
        df = pd.DataFrame(columns=['feature', 'importance'])
        df['feature'] = self.feature_names
        df['importance'] = importance
        df.sort_values(by='importance', axis=0, ascending=False,
                       ignore_index=True).to_excel('./tmp/result/{}'.format(self.date_flag) +
                                                   '/{}_feature_importance.xlsx'.format(self.tree_mode),
                                                   index=False)

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
        create_dirs('./tmp/result/{}'.format(self.date_flag))
        plt.savefig('./tmp/result/{}/{}_feature_importance.png'.format(self.date_flag, self.tree_mode))
        self._importance_export(importance)
        print('特征重要性图以及重要性排序结果文件生成完毕')

    def train(self):
        """
        树模型训练。先进行贝叶斯调参，然后用调整好的参数进行训练并输出模型。
        :return:
        """
        if self.manual_params is None:
            best_params = self._hyperparam_search()
            if self.tree_mode is 'RF':
                if best_params['criterion'] == 0:
                    best_params['criterion'] = 'mse'
                elif best_params['criterion'] == 1:
                    best_params['criterion'] = 'friedman_mse'
                elif best_params['criterion'] == 2:
                    best_params['criterion'] = 'mae'

            print('贝叶斯优化的最优参数组合是:', best_params)
            # 构建树模型训练
            _, model = self._tree_training(best_params)
        else:
            _, model = self._tree_training(self.manual_params)
        return model

    def evaluate(self, model):
        """
        对训练好的模型进行评估，评估其数值差异，涨跌准确率。
        :param model:
        :return:
        """
        train_pred = model.predict(self.x_train)
        test_pred = model.predict(self.x_test)
        self._fitting_plot(train_pred, test_pred)
        print('-----------------------')
        print('训练误差：{}，验证误差：{}'.format(regression_evaluate(self.y_train, train_pred),
                                       regression_evaluate(self.y_test, test_pred)))

        df_eval = up_down_acc(self.y_train, train_pred, self.y_test, test_pred)
        print('训练({})涨跌准确率：{}，验证({})涨跌准确率：{}'.format(len(self.y_train),
                                                     df_eval['acc_train'][0],
                                                     len(self.y_test),
                                                     df_eval['acc_test'][0]))
        # 特征重要性
        self._feature_importance(model)
        return df_eval['acc_test'][0]


class NeuralNetworkPredict(ModelBase):
    """
    神经网络模型。
    """
    def __init__(self, x, y, feature_names):
        self.x_data = x
        self.y_data = y
        self.feature_names = feature_names

    def _preprocessing(self):
        pass

    def train(self):
        pass

    def evaluate(self, model):
        pass
