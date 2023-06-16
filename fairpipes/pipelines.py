import pandas as pd
import sklearn
import seaborn as sns
from sklearn.utils.validation import check_is_fitted
import tensorflow as tf
from fairpipes.operators import column_shuffle, column_killing, redistribution, new_class
from sklearn.metrics import accuracy_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, false_negative_rate, \
    true_negative_rate
import matplotlib.pyplot as plt
from tqdm import tqdm
from fairpipes.models import TensorflowModel, SklearnModel


def basic_fairness_assessment(
        model,
        X_test,
        y_test,
        protected_attributes,
        mutation_ratio=0,
        tol=0.1,
        nb_iter=100
) -> tuple:
    """
    Basic fairness pipeline.

    Make X_test from fairpipes.operators column_shuffle, column_dropping, redistribution, new_class mutants

    Compute model predictions for X_test and mutants.

    Compute metrics from fairlearn.metrics demographic_parity_difference, equalized_odds_ratio, false_negative_rate,
    true_negative_rate

    Plot results and return them

    :param model: an instance of sklearn.base.BaseEstimator or tensorflow.keras.Model
        The model should be already trained to allow predictions in the pipeline.

    :param X_test: a pandas DataFrame without the target variable.

    :param y_test: a pandas Series corresponding to the target

    :param protected_attributes: str
        protected attribute column name

    :param mutation_ratio: float, default 0.0
        percentage of data to mutate in X_test

    :param tol: float, default 0.1
        miss-shuffling ratio tolerance

    :param nb_iter: int, default 100
        number of pipeline iterations to make p-value tests

    :return: python tuple, dict / dict
        tuple with :

            [1]: Dictionary with protected attributes as keys and assessment in values

            [2]: Dictionary with metrics values for each operator and each iteration
    """
    assert type(X_test) == pd.DataFrame, \
        'Type error: X_test should be an instance of <pandas.core.frame.DataFrame>'
    assert isinstance(y_test, pd.Series), 'Type error: target elements should be an instance of <pandas.core.Series>'
    assert isinstance(protected_attributes, list), 'Type error: protected_attributes should be an instance of list'
    for attribute in protected_attributes:
        assert isinstance(attribute, str), 'Type error: protected_attributes elements should be an instance of str'
        assert attribute in X_test.columns, \
            f'Key error: {attribute} not found in dataFrame columns'
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

    history = {'original': {'accuracy': [], 'dpd': [], 'eod': [], 'fnr': [], 'tnr': []},
                     'column_shuffle': {'accuracy': [], 'dpd': [], 'eod': [], 'fnr': [], 'tnr': []},
                     'column_dropping': {'accuracy': [], 'dpd': [], 'eod': [], 'fnr': [], 'tnr': []},
                     'redistribution': {'accuracy': [], 'dpd': [], 'eod': [], 'fnr': [], 'tnr': []},
                     'new_class': {'accuracy': [], 'dpd': [], 'eod': [], 'fnr': [], 'tnr': []}}
    names = list(history.keys())

    for _ in tqdm(range(nb_iter)):
        results = {}

        df_shuffle = column_shuffle(X_test, attribute, mutation_ratio, tol)
        df_dropped = column_killing(X_test, attribute)
        df_redistributed = redistribution(X_test, attribute)
        df_new_class = new_class(X_test, attribute, mutation_ratio)
        mutants = [X_test, df_shuffle, df_dropped, df_redistributed, df_new_class]
        predictions = model.predict(mutants)

        metric_dfs = []
        for i in range(len(predictions)):
            acc = accuracy_score(y_test, predictions[i])
            dpd = demographic_parity_difference(y_true=y_test, y_pred=predictions[i],
                                                sensitive_features=X_test[attribute].values)
            eod = equalized_odds_difference(y_true=y_test, y_pred=predictions[i],
                                            sensitive_features=X_test[attribute].values)
            fnr = false_negative_rate(y_true=y_test, y_pred=predictions[i])
            tnr = true_negative_rate(y_true=y_test, y_pred=predictions[i])

            metrics = {'name': names[i],
                       'accuracy': acc,
                       'dpd': dpd,
                       'eod': eod,
                       'fnr': fnr,
                       'tnr': tnr}

            history[names[i]]['accuracy'].append(acc)
            history[names[i]]['dpd'].append(dpd)
            history[names[i]]['eod'].append(eod)
            history[names[i]]['fnr'].append(fnr)
            history[names[i]]['tnr'].append(tnr)

            metric_df = pd.DataFrame(metrics, index=[0])
            metric_dfs.append(metric_df)

        result_df = pd.concat(metric_dfs, ignore_index=True)
        results[attribute] = result_df

    return results, history
