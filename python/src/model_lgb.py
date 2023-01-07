import lightgbm as lgb
import matplotlib.pyplot as plt
from model import Model
import numpy as np
import pandas as pd
import pickle


class ModelLightGB(Model):
    def __create_model(self):
        """
        create model
        """
        pass

    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        """
        train model
        """
        lgb_trains = lgb.Dataset(X_train, y_train)

        validation = (X_valid is not None) and (y_valid is not None)
        if validation:
            lgb_valids = lgb.Dataset(X_valid, y_valid)

        # set hyper parameters
        params = dict(self.params)
        verbose_eval = params.pop('verbose_eval')
        num_boost_round = params.pop('num_boost_round')

        if validation:
            early_stopping_rounds = params.pop('early_stopping_rounds')
            # start training and validation
            self.model = lgb.train(
                params=params, train_set=lgb_trains, valid_sets=(lgb_trains, lgb_valids),
                num_boost_round=num_boost_round,
                callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=verbose_eval),
                           lgb.log_evaluation(verbose_eval)]
            )
        else:
            # start training
            # ここは実際に使われない必要性を要検討!!!!
            self.model = lgb.train(
                params=params, train_set=lgb_trains, num_boost_round=num_boost_round
            )
        pass

    def predict(self, X_test):
        """
        predict test
        """
        y_pred = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        return y_pred
    
    def save(self, model_dir):
        """
        save model to pickle file
        """
        pickle.dump(self.model, open(model_dir + self.run_name + f"-{self.n_fold}.model", 'wb'))

    def load(self, model_dir):
        """
        load model from pickle file
        """
        self.model = pickle.load(open(model_dir + self.run_name + f"-{self.n_fold}.model", 'rb'))
        pass

    def importance(self, model_dir=None):
        """
        importance
        save gain and split impotance figures if set model directory name
        """
        if model_dir:
            lgb.plot_importance(self.model, figsize=(12, 8), max_num_features=50, importance_type='gain')
            plt.tight_layout()
            plt.savefig(model_dir + self.run_name + f"-gain-{self.n_fold}.png")

            lgb.plot_importance(self.model, figsize=(12, 8), max_num_features=50, importance_type='split')
            plt.tight_layout()
            plt.savefig(model_dir + self.run_name + f"-split-{self.n_fold}.png")

        gain = self.model.feature_importance(importance_type='gain')
        split = self.model.feature_importance(importance_type='split')
        return gain, split
    