import pandas as pd
from tqdm import tqdm

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted

import tensorflow as tf

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

    # encapsulate model
    if isinstance(model, sklearn.base.BaseEstimator):
        model = SklearnModel(model)

    else:
        model = TensorflowModel(model, y_test.unique())

    # prepare empty history
    history = {'original': {'accuracy': [], 'dpd': [], 'eod': []},
               'column_shuffle': {'accuracy': [], 'dpd': [], 'eod': []},
               'column_killing': {'accuracy': [], 'dpd': [], 'eod': []},
               'redistribution': {'accuracy': [], 'dpd': [], 'eod': []},
               'new_class': {'accuracy': [], 'dpd': [], 'eod': []}}
    names = list(history.keys())

    # run nb_iter times the pipeline
    for _ in tqdm(range(nb_iter)):
        # make mutants datasets
        df_shuffle = column_shuffle(X_test, protected_attribute, mutation_ratio)
        df_dropped = column_killing(X_test, protected_attribute)
        df_redistributed = redistribution(X_test, protected_attribute)
        df_new_class = new_class(X_test, protected_attribute, mutation_ratio)

        # compute model prediction for all mutants
        mutants = [X_test, df_shuffle, df_dropped, df_redistributed, df_new_class]
        predictions = model.predict(mutants)

        # store in history
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
        nb_iter=100
) -> list:
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

    :return: list
        histories of each mutation ratio assessment
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

    return histories
