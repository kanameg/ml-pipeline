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


# def get_feature_logger(log_filename:str) -> logging.Logger:
#     """
#     setting logger
#     """
#     # setteing log stream handler
#     stream_handler = StreamHandler()
#     stream_handler.setLevel(DEBUG)
#     stream_handler.setFormatter(Formatter("%(message)s"))
#     # settting log file handler
#     file_handler = FileHandler(
#         log_filename
#     )
#     file_handler.setLevel(DEBUG)
#     file_handler.setFormatter(
#         Formatter("%(asctime)s [%(levelname)7s] (%(name)s) %(message)s")
#     )
#     # setting root logger
#     logging.basicConfig(level=NOTSET, handlers=[stream_handler, file_handler])

#     # create logger
#     logger = logging.getLogger(__name__)

#     logger.info("-"*100)
#     logger.info("start creating feature values")

#     return logger


"""
basic class for getting feature parameter
"""
class Feature(metaclass=ABCMeta):
    """
    basic class for getting feature parameter
    """

    dir = '../data/features/' # default directroy

    def __init__(self, train=None, test=None) -> None:
        # self.logger = logging.getLogger(self.__class__.__name__)
        self.logger = Logger('feature')

        self.name = re.sub("([A-Z])", lambda m: "_" + m.group(1).lower(), self.__class__.__name__).lstrip('_')
        self.logger.debug(f"{self.__class__.__name__} -> {self.name}")

        self.train = train
        self.test = test

        self.train_csv_path = Path(self.dir) / f"{self.name}_train.csv"
        self.test_csv_path = Path(self.dir) / f"{self.name}_test.csv"

        # if csv files exist, load the features from CSV files
        if self.check_csv():
            self.feature_train = pd.DataFrame()
            self.feature_test = pd.DataFrame()
            self.load_csv()
        else:
            self.feature_train = pd.DataFrame()
            self.feature_test = pd.DataFrame()
    
    @contextmanager
    def timer(self):
        """
        timer for mesuring execute time
        """ 
        t0 = time.time()
        self.logger.info(f"[{self.name}] start!")
        yield
        self.logger.info(f"[{self.name}] done in {time.time() - t0:.3f} s")

    @abstractmethod
    def create_feature(self):
        """
        abstract method for creating feature values
        """
        raise NotImplementedError

    def run(self):
        """
        runner method for creating feature
        """
        with self.timer():
            self.create_feature()
        return self
    
    def save_csv(self):
        """
        save the feature for CSV format
        """
        self.feature_train = self.feature_train.reset_index(drop=True)
        self.feature_test = self.feature_test.reset_index(drop=True)
        self.feature_train.to_csv(str(self.train_csv_path), index=False)
        self.feature_test.to_csv(str(self.test_csv_path), index=False)
        return self

    def load_csv(self):
        """
        load the feature from CSV format
        """
        self.feature_train = pd.read_csv(str(self.train_csv_path))
        self.feature_test = pd.read_csv(str(self.test_csv_path))
            
    def check_csv(self):
        """
        check for the csv file existence
        """
        if self.train_csv_path.exists() and self.test_csv_path.exists():
            return True
        else:
            return False

    def create_memo(self, message):
        """
        add feature memo to csv file
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