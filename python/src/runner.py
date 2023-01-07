# -------------------------------------------------------------------
# Training and Test Pipe Line
# -------------------------------------------------------------------
import json
from logger import Logger
import matplotlib.pyplot as plt
from model_lgb import ModelLightGB
import numpy as np
import os
import pandas as pd
import re
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold
from sklearn.metrics import mean_absolute_error
import yaml


# -------------------------------------------------------------------
# read configuration YAML file
# -------------------------------------------------------------------
CONFIG_FILE = '../configs/config.yaml'

with open(CONFIG_FILE) as file:
    yml = yaml.safe_load(file)

LOG_DIR = yml['SETTING']['LOG_DIR']
MODEL_DIR = yml['SETTING']['MODEL_DIR']
FEATURE_DIR = yml['SETTING']['FEATURE_DIR']
RAW_DATA_DIR = yml['SETTING']['RAW_DATA_DIR']
SUBMISSION_DIR = yml['SETTING']['SUBMISSION_DIR']


class Runner:
    """
    pipeline runner class
    """
    def __init__(self, run_name, target, features, params_compe, params_cv, params_model, feature_dir=FEATURE_DIR, model_dir=MODEL_DIR, log_dir=LOG_DIR, raw_data_dir=RAW_DATA_DIR, submission_dir=SUBMISSION_DIR) -> None:
        """
        initialize pipeline runner
        """
        # self.logger = logging.getLogger(self.__class__.__name__) 
        self.logger = Logger(run_name=run_name)
        self.run_name = run_name

        # store feature and target parameters
        self.features = self.__get_feature_name(features) # feature values
        self.target = self.__get_target_name(target)  # target values

        # store parameters
        self.params_cv = params_cv
        self.params_model = params_model
        self.params_compe = params_compe

        # report file
        self.report_name = log_dir + self.run_name + '-report.md'

        # store cross validation parameters
        self.cv_method = params_cv.get('method')
        self.cv_n_splits = params_cv.get('n_splits')
        self.cv_random_state = params_cv.get('random_state')
        self.cv_shuffle = params_cv.get('shuffle')
        self.cv_target = params_cv.get('target')

        # store directory settings
        self.feature_dir = feature_dir
        self.model_dir = model_dir + self.run_name + '/'
        os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)
        self.log_dir = log_dir + self.run_name + '/'
        os.makedirs(os.path.dirname(self.log_dir), exist_ok=True)
        self.raw_data_dir = raw_data_dir
        self.submission_dir = submission_dir

        # load training and test data
        self.x_train = self.__load_x_train()
        self.y_train = self.__load_y_train()
        self.x_test = self.__load_x_test()


    def __get_feature_name(self, features):
        """
        get column name from feature class name
        """
        return [re.sub("([A-Z])", lambda m: "_" + m.group(1).lower(), f).lstrip('_') for f in features]


    def __get_target_name(self, target):
        """
        get column name from target class name
        """
        return re.sub("([A-Z])", lambda m: "_" + m.group(1).lower(), target).lstrip('_')


    def __load_x_train(self):
        """
        load training feature dataset from feature directory
        """
        self.logger.info(f"start loading feature datasets for training")

        dfs = [pd.read_csv(self.feature_dir + f"{fn}_train.csv") for fn in self.features]
        X_train = pd.concat(dfs, axis=1)

        self.logger.info(f"end loading feature datasets for training")
        return X_train


    def __load_y_train(self):
        """
        load training target data from feature directory
        """
        self.logger.info(f"start loading target data for training")

        y_train = pd.read_csv(self.feature_dir + f"{self.target}_train.csv")

        self.logger.info(f"end loading taget data for training")
        return y_train


    def __load_x_test(self):
        """
        load testing fature dataset from feature directory
        """
        self.logger.info(f"start loading feature datasets for testing")

        dfs = [pd.read_csv(self.feature_dir + f"{f}_test.csv") for f in self.features]
        X_test = pd.concat(dfs, axis=1)

        self.logger.info(f"end loading feature datasets for testing")
        return X_test

    def __save_importance_figure(self, importance, type='split'):
        """
        save feature importance to image file
        """
        agg = pd.DataFrame()
        agg['mean'] = importance.mean()
        agg['std'] = importance.std()
        agg['cov'] = agg['std']/agg['mean']
        agg.sort_values(['mean'], ascending=False, inplace=True)

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.bar(x=agg.index, height=agg['mean'], color='y', label='mean', width=0.5)
        ax1.bar(x=agg.index, height=agg['std'], color='orange', label='std', width=0.5)
        ax2.plot(agg['cov'], color='r', marker='o', label='cov')
        ax2.grid(False)
        fig.autofmt_xdate(rotation=45)
        fig.set_figwidth(12)
        fig.set_figheight(6)
        plt.title(f'Feature Importance ({type})', fontsize=14)
        fig.tight_layout()
        fig.savefig(self.model_dir + self.run_name + '-' + type + '.png')

    def sava_parameters(self):
        """
        save model parameter, cv parameter and using features
        """
        params = {}
        params['compe'] = self.params_compe
        params['cv'] = self.params_cv
        params['model'] = self.params_model
        params['features'] = self.features

        f = open(self.model_dir + self.run_name +'-params.json', 'w')
        json.dump(params, f, indent=2, ensure_ascii=False)
        f.close()
        pass

    
    def train_cv(self):
        """
        CV create and save model by cross validation method
        ※ 処理がモデルのライブラリに依存しているのでmodelクラスを作成してscikit-leanでもlightGBMでも処理できるように変更必要
        """
        self.logger.info(f"start cross validation")

        maes = []
        train_idxs = []
        valid_idxs = []
        preds = []

        gains = pd.DataFrame(columns=self.features)
        splits = pd.DataFrame(columns=self.features)
        
        if self.cv_method is 'KFold':
            self.logger.info(f"cv method is KFold method.")
            kfold = KFold(n_splits=self.cv_n_splits, shuffle=self.cv_shuffle, random_state=self.cv_random_state).split(self.x_train, self.y_train)
        elif self.cv_method is 'StratifiedKFold':
            self.logger.info(f"cv method is StratifiedKFold method.")
            kfold = StratifiedKFold(n_splits=self.cv_n_splits, shuffle=self.cv_shuffle, random_state=self.cv_random_state).split(self.x_train, self.x_train[self.cv_target])
        
        for fold_i, (train_index, valid_index) in enumerate(kfold):
            self.logger.info(f"fold {fold_i} start")

            # -------------------------------------------------------------------
            # split training and validation dataset
            # -------------------------------------------------------------------
            X_train = self.x_train.iloc[train_index] # explanatory dataset for training
            y_train = self.y_train.iloc[train_index] # target data for training
            X_valid = self.x_train.iloc[valid_index] # explanatory dataset for valaidation
            y_valid = self.y_train.iloc[valid_index] # target data for validation

            # -------------------------------------------------------------------
            # model training
            # -------------------------------------------------------------------
            model = ModelLightGB(self.run_name, fold_i, self.params_model)

            self.logger.info(f"fold {fold_i} start training model")
            model.train(X_train, y_train, X_valid, y_valid)
            # -------------------------------------------------------------------
            # predict by trained model
            # -------------------------------------------------------------------
            self.logger.info(f"fold {fold_i} start prediction")
            y_valid_pred = model.predict(X_valid)
            
            # -------------------------------------------------------------------
            # caliculate Mean Absolute Error(MAE)
            # -------------------------------------------------------------------
            self.logger.info(f"fold {fold_i} start calculating score")
            mae = mean_absolute_error(y_valid, y_valid_pred)
            self.logger.info(f"fold {fold_i}    calculating score --> MAE: {mae}")

            # -------------------------------------------------------------------
            # save trained model
            # -------------------------------------------------------------------
            self.logger.info(f"fold {fold_i} start saveing trained model")
            model.save(self.model_dir)

            # -------------------------------------------------------------------
            # save model importance
            # -------------------------------------------------------------------
            self.logger.info(f"fold {fold_i} start saveing model importance")
            gain, split = model.importance()
            gains = pd.concat([gains, pd.DataFrame(gain, index=self.features).T], axis=0)
            splits = pd.concat([splits, pd.DataFrame(split, index=self.features).T], axis=0)

            # -------------------------------------------------------------------
            # print training infomations
            # -------------------------------------------------------------------
            train_idxs.append(train_index)
            valid_idxs.append(valid_index)
            preds.append(y_valid_pred)
            maes.append(mae)
        
        # -------------------------------------------------------------------
        # output feature importance
        # -------------------------------------------------------------------
        gains.to_csv(self.model_dir + self.run_name + f"-gain.csv", index=False)
        self.__save_importance_figure(gains, 'gain')

        splits.to_csv(self.model_dir + self.run_name + f"-split.csv", index=False)
        self.__save_importance_figure(splits, 'split')

        # -------------------------------------------------------------------
        # output training result score
        # -------------------------------------------------------------------
        scores = pd.DataFrame()
        scores['MAE'] = maes
        scores.to_csv(self.model_dir + self.run_name + '-score.csv', index=False)

        train_indexs = pd.DataFrame(train_idxs)
        train_indexs.to_csv(self.model_dir + self.run_name + '-train-idex.csv', index=False)
        valid_indexs = pd.DataFrame(train_idxs)
        valid_indexs.to_csv(self.model_dir + self.run_name + '-valid-idex.csv', index=False)
        
        self.logger.info(f"end cross validation")


    def predict_cv(self):
        """
        predict test dataset by trained models
        """
        self.logger.info(f"start predict with test dataset by cross validation models")
        X_test = self.__load_x_test()

        preds = []
        for fold_i in range(self.cv_n_splits):
            self.logger.info(f"fold {fold_i} start prediction")
            model = ModelLightGB(self.run_name, fold_i, self.params_model)
            model.load(self.model_dir)
            pred = model.predict(X_test)
            preds.append(pred)

        pred_mean = np.mean(preds, axis=0)

        df = pd.DataFrame(pred_mean)
        df.to_csv(self.model_dir + self.run_name + '-pred.csv', index=False)

        self.logger.info(f"end predicting with test dataset by cross validation models")
        

    def create_submission(self):
        """
        create submission data from prediction data
        """
        self.logger.info(f"start creating submission data")

        submission = pd.read_csv(self.raw_data_dir + self.params_compe.get('submission_sample'))
        pred = pd.read_csv(self.model_dir + self.run_name + '-pred.csv')
        submission[self.params_compe.get('submission_target')] = pred
        submission.to_csv(self.submission_dir + f"{self.run_name}-submission.csv", index=False)

        self.logger.info(f"end creating submission data")