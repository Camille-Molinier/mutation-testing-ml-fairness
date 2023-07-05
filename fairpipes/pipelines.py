import pandas as pd

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted

import tensorflow as tf

from fairpipes.utils import hist_to_dataframe
from fairpipes.models import TensorflowModel, SklearnModel
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairpipes.operators import column_shuffle, column_killing, redistribution, new_class


def basic_fairness_assessment(
        model,
        X_test,
        y_test,
        protected_attribute,
        mutation_ratio=0,
        nb_iter=100
) -> dict:
    """
    Basic fairness pipeline.

    Make X_test from fairpipes.operators column_shuffle, column_dropping, redistribution, new_class mutants

    Compute model predictions for X_test and mutants.

    Compute metrics from fairlearn.metrics demographic_parity_difference, equalized_odds_ratio, false_negative_rate,
    true_negative_rate

    :param model: an instance of sklearn.base.BaseEstimator or tensorflow.keras.Model
        The model should be already trained to allow predictions in the pipeline.

    :param X_test: a pandas DataFrame without the target variable.

    :param y_test: a pandas Series corresponding to the target

    :param protected_attribute: str
        protected attribute column name

    :param mutation_ratio: float, default 0.0
        percentage of data to mutate in X_test

    :param nb_iter: int, default 100
        number of pipeline iterations to make p-value tests

    :return: python dictionary with operators names as keys and metrics distributions as values
    """
    assert type(X_test) == pd.DataFrame, \
        'Type error: X_test should be an instance of <pandas.core.frame.DataFrame>'
    assert isinstance(y_test, pd.Series), 'Type error: target elements should be an instance of <pandas.core.Series>'
    assert isinstance(protected_attribute, str), 'Type error: protected_attributes should be an instance of str'
    assert protected_attribute in X_test.columns, f'Key error: {protected_attribute} not found in dataFrame columns'
    assert isinstance(mutation_ratio, float), 'Type error: mutation_ratio elements should be an instance of float'
    assert isinstance(nb_iter, int), 'Type error: nb_iter elements should be an instance of int'
    assert isinstance(model, sklearn.base.BaseEstimator) or isinstance(model, tf.keras.Model), \
        'Type error: model elements should be an instance of <sklearn.base.BaseEstimator> or <tensorflow.keras.Model>'

    if isinstance(model, sklearn.base.BaseEstimator):
        check_is_fitted(model)
        model = SklearnModel(model)

    else:
        if model.history is None:
            raise AssertionError('Tensorflow model should be fitted')
        model = TensorflowModel(model, y_test.unique())

    history = {'original': {'accuracy': [], 'dpd': [], 'eod': []},
               'column_shuffle': {'accuracy': [], 'dpd': [], 'eod': []},
               'column_dropping': {'accuracy': [], 'dpd': [], 'eod': []},
               'redistribution': {'accuracy': [], 'dpd': [], 'eod': []},
               'new_class': {'accuracy': [], 'dpd': [], 'eod': []}}
    names = list(history.keys())

    for _ in range(nb_iter):

        df_shuffle = column_shuffle(X_test, protected_attribute, mutation_ratio)
        df_dropped = column_killing(X_test, protected_attribute)
        df_redistributed = redistribution(X_test, protected_attribute)
        df_new_class = new_class(X_test, protected_attribute, mutation_ratio)

        mutants = [X_test, df_shuffle, df_dropped, df_redistributed, df_new_class]
        predictions = model.predict(mutants)

        for i in range(len(predictions)):
            acc = accuracy_score(y_test, predictions[i])
            dpd = demographic_parity_difference(y_true=y_test, y_pred=predictions[i],
                                                sensitive_features=X_test[protected_attribute].values)
            eod = equalized_odds_difference(y_true=y_test, y_pred=predictions[i],
                                            sensitive_features=X_test[protected_attribute].values)

            history[names[i]]['accuracy'].append(acc)
            history[names[i]]['dpd'].append(dpd)
            history[names[i]]['eod'].append(eod)

    return history


def multi_mutation_fairness_assessment(
        model,
        X_test,
        y_test,
        protected_attribute,
        mutation_ratios,
        nb_iter=100,
        output_name='out'
) -> tuple:
    """
    Multi mutation ratio fairness pipeline.

    Make a simple mutation pipeline for each mutation ratio in parameter

    :param model: an instance of sklearn.base.BaseEstimator or tensorflow.keras.Model
        The model should be already trained to allow predictions in the pipeline.

    :param X_test: a pandas DataFrame without the target variable.

    :param y_test: a pandas Series corresponding to the target

    :param protected_attribute: str
        protected attribute column name

    :param mutation_ratios: list
        data percentages to mutate in X_test

    :param nb_iter: int, default 100
        number of pipeline iterations to make p-value tests

    :param output_name: str, default 'out'
        name of the output csv file

    :return: python tuple, list / list
        tuple with :

            [1]: List of dictionaries with protected attributes as keys and assessment in values

            [2]: List of dictionaries with metrics values for each operator and each iteration
    """
    assert type(X_test) == pd.DataFrame, \
        'Type error: X_test should be an instance of <pandas.core.frame.DataFrame>'
    assert isinstance(y_test, pd.Series), 'Type error: target elements should be an instance of <pandas.core.Series>'
    assert isinstance(protected_attribute, str), 'Type error: protected_attributes should be an instance of str'
    assert isinstance(mutation_ratios, list), 'Type error: mutation_ratios should be an instance of list'
    assert isinstance(nb_iter, int), 'Type error: nb_iter elements should be an instance of int'
    assert isinstance(model, sklearn.base.BaseEstimator) or isinstance(model, tf.keras.Model), \
        'Type error: model elements should be an instance of <sklearn.base.BaseEstimator> or <tensorflow.keras.Model>'

    histories = []

    for i in range(len(mutation_ratios)):
        hist = basic_fairness_assessment(model, X_test, y_test, protected_attribute,
                                         mutation_ratio=mutation_ratios[i], nb_iter=nb_iter)
        histories.append(hist)

    result_df = hist_to_dataframe(histories)
    result_df.to_csv(f'./dat/exports/{output_name}.csv')

    return histories
