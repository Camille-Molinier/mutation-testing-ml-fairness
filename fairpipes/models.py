from abc import ABC, abstractmethod

import sklearn
import numpy as np
import tensorflow as tf


########################################################################################################################
#                                                      Interface                                                       #
########################################################################################################################
class Model(ABC):
    @abstractmethod
    def predict(self, dataframes=[]) -> list:
        """
        Make predictions for a list of DataFrames

        :param dataframes: list, default []
            pandas DataFrames list to predictions
        :return: list of pandas.Series containing predictions in the same order as dataframes
        """
        pass


########################################################################################################################
#                                                    Sklearn model                                                     #
########################################################################################################################
class SklearnModel(Model):
    def __init__(self, model):
        assert isinstance(model, sklearn.base.BaseEstimator)
        self.model = model

    def predict(self, dataframes=[]) -> list:
        predictions = []

        for df in dataframes:
            predictions.append(self.model.predict(df))

        return predictions


########################################################################################################################
#                                                   Tensorflow model                                                   #
########################################################################################################################
class TensorflowModel(Model):
    def __init__(self, model, classes, predict_function):
        assert isinstance(model, tf.keras.Model)
        self.model = model
        self.classes = classes
        self.simple_predict = predict_function

    def predict(self, dataframes=[]) -> list:
        predictions = []

        for df in dataframes:
            predictions.append([self.classes[np.argmax(prediction)] for prediction in self.simple_predict(df.values)])

        return predictions
