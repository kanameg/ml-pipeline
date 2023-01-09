# -------------------------------------------------------------------
# ライブラリの読込
# -------------------------------------------------------------------
from abc import ABCMeta, abstractmethod
import category_encoders as ce
from contextlib import contextmanager
# import logging
# from logging import StreamHandler, FileHandler, Formatter
# from logging import INFO, DEBUG, NOTSET
from logger import Logger
import os
import pandas as pd
from pathlib import Path
import re
import time


"""
特徴量を取得するための基本クラス
"""
class Feature(metaclass=ABCMeta):
    """
    特徴量の基本クラス
    """

    dir = '../data/features/' # 保存ベースディレクトリ

    def __init__(self, train=None, test=None) -> None:
        self.logger = Logger('feature')

        self.name = re.sub("([A-Z])", lambda m: "_" + m.group(1).lower(), self.__class__.__name__).lstrip('_')
        self.logger.debug(f"{self.__class__.__name__} -> {self.name}")

        # データを結合して保存
        self.all = pd.concat(train, test, sort=False).reset_index(drop=True)

        # 特徴量保存パス
        self.train_csv_path = Path(self.dir) / f"{self.name}_train.csv"
        self.test_csv_path = Path(self.dir) / f"{self.name}_test.csv"

        # 既にcsvファイルが存在すればcsvを特徴量として読み込む
        if self.is_exist_csv():
            self.feature_train = pd.DataFrame()
            self.feature_test = pd.DataFrame()
            self.load_csv()
        else:
            self.feature_train = pd.DataFrame()
            self.feature_test = pd.DataFrame()
    
    @contextmanager
    def timer(self):
        """
        実行時間取得用Timer
        """ 
        t0 = time.time()
        self.logger.info(f"[{self.name}] start!")
        yield
        self.logger.info(f"[{self.name}] done in {time.time() - t0:.3f} s")

    @abstractmethod
    def create_feature(self):
        """
        特徴量生成
        """
        raise NotImplementedError

    def run(self):
        """
        特徴量生成実行
        """
        with self.timer():
            self.create_feature()
        return self
    
    def save_csv(self):
        """
        CSVへ特徴量を保存
        """
        self.feature_train = self.feature_train.reset_index(drop=True)
        self.feature_test = self.feature_test.reset_index(drop=True)
        self.feature_train.to_csv(str(self.train_csv_path), index=False)
        self.feature_test.to_csv(str(self.test_csv_path), index=False)
        return self

    def load_csv(self):
        """
        CSVから特徴量を読み出し
        """
        self.feature_train = pd.read_csv(str(self.train_csv_path))
        self.feature_test = pd.read_csv(str(self.test_csv_path))
            
    def is_exist_csv(self):
        """
        CSVファイルが既に存在するかチェック
        """
        if self.train_csv_path.exists() and self.test_csv_path.exists():
            return True
        else:
            return False

    def create_memo(self, message):
        """
        特徴量のメモを生成
        """
        file_path = self.dir + '_feature_memo.csv'
        if not os.path.exists(file_path):
            pd.DataFrame([], columns=['name', 'memo']).to_csv(file_path, index=False)
        
        memo = pd.read_csv(file_path)
        name = self.__class__.__name__
        # check this feature is already in csv file
        if name not in memo['name']:
            memo = pd.concat([memo, pd.DataFrame([[name, message]], columns=memo.columns)], axis=0)
            memo.to_csv(file_path, index=False)
        