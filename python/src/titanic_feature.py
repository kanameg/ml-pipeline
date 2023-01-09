# -------------------------------------------------------------------
# create feature of second hand apartments
# -------------------------------------------------------------------
import argparse
import category_encoders as ce
from contextlib import contextmanager
from feature import Feature
import glob
from logger import Logger
import numpy as np
import os
import pandas as pd
import time
import yaml


# -------------------------------------------------------------------
# read configuration YAML file
# -------------------------------------------------------------------
CONFIG_FILE = '../configs/config.yaml'

with open(CONFIG_FILE) as file:
    yml = yaml.safe_load(file)

LOG_DIR = yml['SETTING']['LOG_DIR']
RAW_DATA_DIR = yml['SETTING']['RAW_DATA_DIR']
FEATURE_DIR = yml['SETTING']['FEATURE_DIR']
Feature.dir = FEATURE_DIR  # set feature file directory to base class


# -------------------------------------------------------------------
# 生成する特徴量
# -------------------------------------------------------------------
features = [
    'PassengerId', 'Pclass', 'Fare', 'FereLog10',
]

class PassengerId(Feature):
    """
    PassengerIdをそのまま利用
    """
    def create_feature(self):        
        self.feature_train[self.name] = self.all.loc[all['train']==True]['PassengerId']
        self.feature_test[self.name] = self.all.loc[all['train']==False]['PassengerId']
        self.create_memo('PassengerIdは乗船者のID 特徴量としては利用しない')

class Pclass(Feature):
    """
    客室等級
    """
    def create_feature(self):
        self.feature_train[self.name] = self.all.loc[all['train']==True]['Pclass']
        self.feature_test[self.name] = self.all.loc[all['train']==False]['Pclass']
        self.create_memo('各乗客の客室等級')
    pass

class Fare(Feature):
    """
    乗船料金
    """
    def create_feature(self):
        self.feature_train[self.name] = self.all.loc[all['train']==True]['Fare']
        self.feature_test[self.name] = self.all.loc[all['train']==False]['Fare']
        self.create_memo('各乗客の乗船料金 欠損値補完済み')
    pass

class FereLog10(Feature):
    """
    乗船料金の対数表現(log10)
    """
    def create_feature(self):
        self.feature_train[self.name] = self.all.loc[all['train']==True]['FareLog10']
        self.feature_test[self.name] = self.all.loc[all['train']==False]['FareLog10']
        self.create_memo('各乗客の乗船料金を対数(log10)表現 欠損値補完済み')
    pass




from abc import ABCMeta, abstractmethod

class PreProcess(metaclass=ABCMeta):
    """
    データ前処理の抽象クラス
    """
    dir = '../data/raw/'

    def __init__(self, train, test) -> None:
        train['train'] = True
        test['train'] = False
        self.all = pd.concat([train, test], axis=0)
        pass
    
    @abstractmethod
    def clean(self):
        """
        欠損値処理
        """
        pass
    
    def save(self):
        """
        前処理済みデータをファイルに保存
        特徴量抽出時に使用する
        """
        train = self.all.loc[self.all['train'] == True]
        test = self.all.loc[self.all['train'] == False]
        # データ保存
        train.to_csv(dir + 'train_prep.csv', index=False)
        test.to_csv(dir + 'test_prep.csv', index=False)
        pass


@contextmanager
def timer(name):
    """
    timer for mesuring execute time
    """ 
    t0 = time.time()
    logger.info(f"[{name}] start!")
    yield
    logger.info(f"[{name}] done in {time.time() - t0:.3f} s")



if __name__ == '__main__':
    # -------------------------------------------------------------------
    # Loggerの設定
    # -------------------------------------------------------------------
    # log_filename = os.path.splitext(__file__)[0] + ".log"
    # logger = get_feature_logger(LOG_DIR + log_filename)
    logger = Logger('feature')

    # -------------------------------------------------------------------
    # 引数を取得 [-f force recreate feature parameter]
    # -------------------------------------------------------------------
    argp = argparse.ArgumentParser(description='default')
    argp.add_argument('--force', '-f', action='store_true', help='Overwrite existing files')
    args = argp.parse_args()
    logger.info(args)

    # -------------------------------------------------------------------
    # 学習データと評価用データの読み込み
    # -------------------------------------------------------------------
    files = []
    files.extend(glob.glob(RAW_DATA_DIR + "train/??.csv"))
    logger.debug(files)
    logger.info(f"number of raw train files: {len(files)}")

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    train = pd.concat(dfs, axis=0, ignore_index=True)
    train['train'] = True
    logger.info(f"train shape = {train.shape}")

    test = pd.read_csv(RAW_DATA_DIR + 'test.csv')
    test['train'] = False
    logger.info(f"test shape = {test.shape}")

    # -------------------------------------------------------------------
    # cleansing raw train and test data
    # -------------------------------------------------------------------
    all = pd.concat([train, test], axis=0)

    train = all.loc[all['train']==True].copy()
    test = all.loc[all['train']==False].copy()

    # -------------------------------------------------------------------
    # create feature parameter in the order of feature array
    # -------------------------------------------------------------------
    for feature in features:
        f = eval(feature)(train, test)
        if f.check_csv() and not args.force:
            print(f.name, 'was skipped')
        else:
            f.run().save_csv()
            f.load_csv()
    