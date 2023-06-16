import pandas as pd
import numpy as np
import random


########################################################################################################################
#                                               Column shuffle operator                                                #
########################################################################################################################
def column_shuffle(dataframe, protected_attribute, shuffle_ratio=0.0, tol=0.1) -> pd.DataFrame:
    """
    Column shuffling operator

    Take a portion of the column of shuffle_ratio*len(dataframe) and randomly make permutations of protected_attributes

    Random permutations can generate unchanged values, tol attribute define unchanged values ratio tolerance.

    :param dataframe: instance of pandas.DataFrame to compute mutation on

    :param protected_attribute: str
        protected attributes column name

    :param shuffle_ratio: float, default 0.0
        ratio of the data to mutate

    :param tol: float, default 0.1
        miss-shuffling ratio tolerance

    :return: pandas.Dataframe
        mutated dataset
    """
    assert type(dataframe) == pd.DataFrame, \
        'Type error: dataframe should be an instance of <pandas.core.frame.DataFrame>'
    assert isinstance(protected_attribute, str), \
        'Type error: protected_attributes elements should be an instance of str'
    assert protected_attribute in dataframe.columns, \
        'Key error: protected_attribute elements should be in dataframe columns'
    assert type(shuffle_ratio) == float, 'Type error: protected_attributes should be an instance of float'
    assert type(tol) == float, 'Type error: protected_attributes should be an instance of float'
    assert 0 <= shuffle_ratio <= 1, 'Value error: shuffle_ratio should be in range(0,1)'
    assert 0 <= tol <= 1, 'Value error: tol should be in range(0,1)'

    # copy dataframe to manipulate it safely
    df = dataframe.copy()

    if shuffle_ratio > 0:

        # compute number of row to shuffle
        n_samples = int(len(df) * shuffle_ratio)
        # choose n_samples random indexes
        random_index = df.sample(n=n_samples).index.tolist()

        # loop to reduce shuffle vanishing
        ratio = 0
        while ratio <= shuffle_ratio - tol:
            # make a permutation of rows
            shuffled = np.random.permutation(df.loc[random_index, protected_attribute])
            # replace in data
            df.loc[random_index, protected_attribute] = shuffled
            # compute shuffle ratio
            ratio = sum(df[protected_attribute] != dataframe[protected_attribute]) / len(dataframe)

    return df


########################################################################################################################
#                                               Column dropping operator                                               #
########################################################################################################################
def column_killing(dataframe, protected_attribute) -> pd.DataFrame:
    """
    Column killing operator

    Set all values in the column with one of the possible values

    Statistically killing the column

    :param dataframe: instance of pandas.DataFrame to compute mutation on

    :param protected_attribute: str
        protected attributes column name

    :return: pandas.Dataframe
        mutated dataset
    """
    assert type(dataframe) == pd.DataFrame, \
        'Type error: dataframe should be an instance of <pandas.core.frame.DataFrame>'
    assert isinstance(protected_attribute, str), \
        'Type error: protected_attributes elements should be an instance of str'
    assert protected_attribute in dataframe.columns, \
        'Key error: protected_attribute elements should be in dataframe columns'

    # make a copy for safe manipulation
    df = dataframe.copy()

    df[protected_attribute] = pd.Series(df[protected_attribute].unique()[0], index=range(len(df)))

    return df


########################################################################################################################
#                                               Redistribution operator                                               #
########################################################################################################################
def redistribution(dataframe, protected_attribute) -> pd.DataFrame:
    """
    Column occurrence redistribution operator

    Compute current value counts and rebalance distribution

    :param dataframe: instance of pandas.DataFrame to compute mutation on

    :param protected_attribute: str
        protected attributes column name

    :return: pandas.Dataframe
        mutated dataset
    """
    assert type(dataframe) == pd.DataFrame, \
        'Type error: dataframe should be an instance of <pandas.core.frame.DataFrame>'
    assert isinstance(protected_attribute, str), \
        'Type error: protected_attributes elements should be an instance of str'
    assert protected_attribute in dataframe.columns, \
        'Key error: protected_attribute elements should be in dataframe columns'

    # make a copy for safe manipulation
    df = dataframe.copy()

    # get column as a series
    df_col = df[protected_attribute]

    # if value are not string, convert to string and store type
    was_not_string = False
    if not isinstance(df_col[0], str):
        was_not_string = True
        old_type = type(df_col[0])
        df_col = df_col.astype(str)

    # compute distribution
    distribution = df_col.value_counts()

    # compute balanced repartition
    redistribution_factor = int(len(df_col) / len(distribution))
    redistribution_leftover = len(df_col) - (redistribution_factor * len(distribution))
    optimal_distribution = pd.Series(redistribution_factor, index=distribution.index)
    optimal_distribution[0] = optimal_distribution[0] + redistribution_leftover

    # loop on rows and mark unchanged values
    unchanged = []
    for i in range(len(df_col)):
        if optimal_distribution[df_col[i]] > 0:
            optimal_distribution[df_col[i]] = optimal_distribution[df_col[i]] - 1
        else:
            unchanged.append(i)

    # compute remaining values
    rest = optimal_distribution[optimal_distribution > 0]
    rest_list = [index for index, value in rest.items() for _ in range(value)]
    random.shuffle(rest_list)

    # place remaining value in unchanged indexes
    copy = df_col.copy()
    copy.loc[unchanged] = rest_list

    # if data was not string, reconvert to ancient type
    if was_not_string:
        copy = copy.astype(old_type)

    # replace dataframe column with balanced version
    df[protected_attribute] = copy

    return df


#################### Begin of unmaintained section ####################

########################################################################################################################
#                                              Duplicate mutant operator                                               #
########################################################################################################################
def duplication_mutation(dataframe, protected_attribute, duplication_ratio=0.0) -> pd.DataFrame:
    """⚠️ Unusable operator"""
    assert type(dataframe) == pd.DataFrame, \
        'Type error: dataframe should be an instance of <pandas.core.frame.DataFrame>'
    assert isinstance(protected_attribute, str), \
        'Type error: protected_attributes elements should be an instance of str'
    assert protected_attribute in dataframe.columns, \
        'Key error: protected_attribute elements should be in dataframe columns'
    assert type(duplication_ratio) == float, 'Type error: protected_attributes should be an instance of float'
    assert 0 <= duplication_ratio <= 1, 'Value error: shuffle_ratio should be in range(0,1)'

    # copy dataframe to manipulate it safely
    df = dataframe.copy()

    if duplication_ratio > 0:

        # compute number of row to shuffle
        n_samples = int(len(df) * duplication_ratio)
        # choose n_samples random indexes
        random_index = df.sample(n=n_samples).index.tolist()

        # get sub dataframe for performance and run through rows
        sub_df = df.loc[random_index]
        for index in random_index:
            # copy column (pandas warning)
            copy = sub_df[protected_attribute].copy()
            # get unique values
            uniques = copy.unique().tolist()
            # remove current value
            uniques.remove(copy[index])
            # replace with random value in uniques
            copy[index] = uniques[np.random.randint(0, len(uniques))]
            # replace column
            sub_df[protected_attribute] = copy

        # add mutant subset to dataframe
        df = pd.concat([df, sub_df])

    return df


#################### End of unmaintained section ####################


########################################################################################################################
#                                                  New class operator                                                  #
########################################################################################################################
def new_class(dataframe, protected_attribute, mutation_ratio=0.0, new_value='null') -> pd.DataFrame:
    """
    New class apparition operator

    Simulate new value in production data for protected attribute

    For numeric columns, new value is max value + 1

    For string columns, new value is 'null' by default, if you have a 'null' class, replace it

    :param dataframe: instance of pandas.DataFrame to compute mutation on

    :param protected_attribute: str
        protected attributes column name

    :param mutation_ratio: float, default 0.0
        proportion of the data to mutate

    :param new_value: str, default 'null'
        new value for str columns

    :return: pandas.Dataframe
        mutated dataset
    """
    assert type(dataframe) == pd.DataFrame, \
        'Type error: dataframe should be an instance of <pandas.core.frame.DataFrame>'
    assert isinstance(protected_attribute, str), \
        'Type error: protected_attributes elements should be an instance of str'
    assert protected_attribute in dataframe.columns, \
        'Key error: protected_attribute elements should be in dataframe columns'
    assert type(mutation_ratio) == float, 'Type error: protected_attributes should be an instance of float'
    assert 0 <= mutation_ratio <= 1, 'Value error: shuffle_ratio should be in range(0,1)'

    # copy dataframe to manipulate it safely
    df = dataframe.copy()

    if mutation_ratio > 0:

        # compute number of row to shuffle
        n_samples = int(len(df) * mutation_ratio)
        # choose n_samples random indexes
        random_index = df.sample(n=n_samples).index.tolist()

        sub_df = df.loc[random_index]
        if isinstance(sub_df[protected_attribute][random_index[0]], str):
            df.loc[random_index, protected_attribute] = new_value

        else:
            df.loc[random_index, protected_attribute] = max(df[protected_attribute]) + 1

    return df
