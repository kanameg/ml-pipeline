import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod

class Model(metaclass=ABCMeta):
    """
    abstruct class for trained model
    """
    def __init__(self, run_name, n_fold, params) -> None:
        self.run_name = run_name
        self.n_fold = n_fold
        self.params = params
        self.model = None
        

    def __create_model(self):
        """
        create model
        """
        pass

    @abstractmethod
    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        """
        train model
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        predict test
        """
        pass

    @abstractmethod
    def save(self):
        """
        save model to pickle file
        """
        pass

    @abstractmethod
    def load(self):
        """
        load model from pickle file
        """
        pass
    