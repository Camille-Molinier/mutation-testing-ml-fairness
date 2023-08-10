from abc import ABC, abstractmethod

import sklearn
import numpy as np
import tensorflow as tf
from sklearn.utils.validation import check_is_fitted


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
        check_is_fitted(model)
        self.model = model

    def predict(self, dataframes=[]) -> list:
        return [self.model.predict(df) for df in dataframes]


########################################################################################################################
#                                                   Tensorflow model                                                   #
########################################################################################################################
class TensorflowModel(Model):
    def __init__(self, model, classes, predict_function=None):
        assert isinstance(model, tf.keras.Model)
        try:
            all([w is not None for w in model.get_weights()])
        except:
            raise AssertionError('Tensorflow model should be fitted')

        self.model = model
        self.classes = classes
        self.simple_predict = predict_function

    def predict(self, dataframes=[]) -> list:
        predictions = []

        for df in dataframes:
            if self.simple_predict is not None:
                predictions.append(
                    [self.classes[np.argmax(prediction)] for prediction in self.simple_predict(df.values)])
            else:
                predictions.append(
                    [self.classes[np.argmax(prediction)] for prediction in self.__simple_predict(df.values)])
        return predictions

    def __simple_predict(self, df):
        y_pred_prob = self.model.predict(df, verbose=0)
        y_pred = (tf.nn.sigmoid(y_pred_prob) > 0.5).numpy()
        y_pred = y_pred.reshape(y_pred.shape[0], ).astype(int)
        return y_pred
