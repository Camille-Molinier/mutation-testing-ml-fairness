import unittest
import random
import numpy as np
import pandas as pd
import sklearn.exceptions

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

from fairpipes.models import TensorflowModel, SklearnModel, Model


########################################################################################################################
#                                                    Abstract model                                                    #
########################################################################################################################
class AbstractModelTest(unittest.TestCase):
    def test_not_usable_class(self):
        with self.assertRaises(TypeError):
            Model()


########################################################################################################################
#                                                    SKLearn model                                                     #
########################################################################################################################
class SKLearnModelTest(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(1)
        data1 = {
            'col1': np.random.randint(1, 100, size=100),
            'col2': np.random.rand(100),
            'col3': np.random.randint(1, 200, size=100)
        }

        data2 = {
            'col1': np.random.rand(100),
            'col2': np.random.randint(1, 100, size=100),
            'col3': np.random.randint(1, 200, size=100)
        }

        self.y1 = [0] * 45 + [1] * 55
        self.y2 = [0] * 55 + [1] * 45
        random.shuffle(self.y1)
        random.shuffle(self.y2)

        # Create dataframes
        self.df1 = pd.DataFrame(data1)
        self.df2 = pd.DataFrame(data2)

    def test_constructor_with_decision_tree(self):
        try:
            SklearnModel(DecisionTreeClassifier().fit(self.df1, self.y1))
        except Exception:
            self.fail("column_shuffle() raised Exception unexpectedly!")

    def test_constructor_with_random_forest(self):
        try:
            SklearnModel(RandomForestClassifier().fit(self.df1, self.y1))
        except Exception:
            self.fail("column_shuffle() raised Exception unexpectedly!")

    def test_constructor_with_svm(self):
        try:
            SklearnModel(SVC().fit(self.df1, self.y1))
        except Exception:
            self.fail("column_shuffle() raised Exception unexpectedly!")

    def test_constructor_with_not_fitted_decision_tree(self):
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            SklearnModel(DecisionTreeClassifier())

    def test_predict_correct_format(self):
        model = SklearnModel(DecisionTreeClassifier().fit(self.df1, self.y1))
        results = model.predict([self.df1, self.df2])
        assert isinstance(results, list)

    def test_predict_correct_format2(self):
        model = SklearnModel(DecisionTreeClassifier().fit(self.df1, self.y1))
        results = model.predict([self.df1, self.df2])
        assert len(results) == 2

    def test_prefictions_correct_format3(self):
        model = SklearnModel(DecisionTreeClassifier().fit(self.df1, self.y1))
        results = model.predict([self.df1])
        assert all(element in np.unique(self.y1) for element in results[0])


########################################################################################################################
#                                                   Tensorflow model                                                   #
########################################################################################################################
class TensorflowModelTest(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(1)
        data1 = {
            'col1': np.random.randint(1, 100, size=100),
            'col2': np.random.rand(100),
            'col3': np.random.randint(1, 200, size=100)
        }

        data2 = {
            'col1': np.random.rand(100),
            'col2': np.random.randint(1, 100, size=100),
            'col3': np.random.randint(1, 200, size=100)
        }

        self.y1 = [0] * 45 + [1] * 55
        self.y2 = [0] * 55 + [1] * 45
        random.shuffle(self.y1)
        random.shuffle(self.y2)

        self.y1 = pd.Series(self.y1)
        self.y2 = pd.Series(self.y2)

        # Create dataframes
        self.df1 = pd.DataFrame(data1)
        self.df2 = pd.DataFrame(data2)

        self.sequential = keras.Sequential(
            [
                Dense(10, activation="relu"),
                Dense(1, name="sigmoid")
            ]
        )
        self.sequential.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.sequential.fit(self.df1, self.y1, epochs=1, verbose=0)
        self.classes = list(np.unique(self.y1))

    def simple_predict(self, df):
        y_pred_prob = self.sequential.predict(df, verbose=0)
        y_pred = (tf.nn.sigmoid(y_pred_prob) > 0.5).numpy()
        y_pred = y_pred.reshape(y_pred.shape[0], ).astype(int)
        return y_pred

    def test_constructor(self):
        try:
            TensorflowModel(self.sequential, self.classes, self.simple_predict)
        except Exception:
            self.fail("column_shuffle() raised Exception unexpectedly!")

    def test_constructor_not_fitted_sequential(self):
        model = keras.Sequential([Dense(10, activation="relu"), Dense(1, name="sigmoid")])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        with self.assertRaises(AssertionError):
            TensorflowModel(model, self.classes, self.simple_predict)

    def test_predict_correct_format(self):
        model = TensorflowModel(self.sequential, self.classes, self.simple_predict)
        results = model.predict([self.df1, self.df2])
        assert isinstance(results, list)

    def test_predict_correct_format2(self):
        model = TensorflowModel(self.sequential, self.classes, self.simple_predict)
        results = model.predict([self.df1, self.df2])
        assert len(results) == 2

    def test_predict_correct_format3(self):
        model = TensorflowModel(self.sequential, self.classes, self.simple_predict)
        results = model.predict([self.df1])
        assert all(element in self.classes for element in results[0])


def make_suite():
    suite = unittest.TestSuite()
    suite.addTest(AbstractModelTest())
    suite.addTest(SKLearnModelTest())
    suite.addTest(TensorflowModelTest())
    return suite


def run_operator_tests():
    runner = unittest.TextTestRunner()
    runner.run(make_suite())


if __name__ == '__main__':
    run_operator_tests()
