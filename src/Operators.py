import pandas as pd
import numpy as np
import random


########################################################################################################################
#                                               Column shuffle operator                                                #
########################################################################################################################
def column_shuffle(dataframe, protected_attributes, shuffle_ratio=0.0, tol=0.1):
    assert type(dataframe) == pd.DataFrame, \
        'Type error: dataframe should be an instance of <pandas.core.frame.DataFrame>'
    assert isinstance(protected_attributes, list), 'Type error: protected_attributes should be an instance of list'
    for attribute in protected_attributes:
        assert isinstance(attribute, str), 'Type error: protected_attributes elements should be an instance of str'
        assert attribute in dataframe.columns, \
            'Key error: protected_attribute elements should be in dataframe columns'
    assert type(shuffle_ratio) == float, 'Type error: protected_attributes should be an instance of float'
    assert type(tol) == float, 'Type error: protected_attributes should be an instance of float'
    assert 0 <= shuffle_ratio <= 1, 'Value error: shuffle_ratio should be in range(0,1)'
    assert 0 <= tol <= 1, 'Value error: tol should be in range(0,1)'

    # copy dataframe to manipulate it safely
    df = dataframe.copy()

    if shuffle_ratio > 0:

        # run through protected attributes columns
        for attribute in protected_attributes:
            # compute number of row to shuffle
            n_samples = int(len(df) * shuffle_ratio)
            # choose n_samples random indexes
            random_index = df.sample(n=n_samples).index.tolist()

            # loop to reduce shuffle vanishing
            ratio = 0
            while ratio <= shuffle_ratio - tol:
                # make a permutation of rows
                shuffled = np.random.permutation(df.loc[random_index, attribute])
                # replace in data
                df.loc[random_index, 'col3'] = shuffled
                # compute shuffle ratio
                ratio = sum(df['col3'] != dataframe['col3']) / len(dataframe)

    return df


########################################################################################################################
#                                               Column dropping operator                                               #
########################################################################################################################
def column_dropping(dataframe, protected_attributes):
    assert type(dataframe) == pd.DataFrame, \
        'Type error: dataframe should be an instance of <pandas.core.frame.DataFrame>'
    assert isinstance(protected_attributes, list), 'Type error: protected_attributes should be an instance of list'
    for attribute in protected_attributes:
        assert isinstance(attribute, str), 'Type error: protected_attributes elements should be an instance of str'
        assert attribute in dataframe.columns, \
            f'Key error: {attribute} not found in dataFrame columns'

    # make a copy for safe manipulation
    df = dataframe.copy()

    # drop all columns
    df.drop(protected_attributes, axis=1, inplace=True)

    return df


########################################################################################################################
#                                               Redistribution operator                                               #
########################################################################################################################
def redistribution(dataframe, protected_attributes):
    assert type(dataframe) == pd.DataFrame, \
        'Type error: dataframe should be an instance of <pandas.core.frame.DataFrame>'
    assert isinstance(protected_attributes, list), 'Type error: protected_attributes should be an instance of list'
    for attribute in protected_attributes:
        assert isinstance(attribute, str), 'Type error: protected_attributes elements should be an instance of str'
        assert attribute in dataframe.columns, \
            f'Key error: {attribute} not found in dataFrame columns'

    # make a copy for safe manipulation
    df = dataframe.copy()

    # run through columns
    for col in protected_attributes:
        # get column as a serie
        df_col = df[col]

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
        df[col] = copy

    return df


########################################################################################################################
#                                              Duplicate mutant operator                                               #
########################################################################################################################
def duplication_mutation(dataframe, protected_attributes, duplication_ratio=0.0):
    assert type(dataframe) == pd.DataFrame, \
        'Type error: dataframe should be an instance of <pandas.core.frame.DataFrame>'
    assert isinstance(protected_attributes, list), 'Type error: protected_attributes should be an instance of list'
    for attribute in protected_attributes:
        assert isinstance(attribute, str), 'Type error: protected_attributes elements should be an instance of str'
        assert attribute in dataframe.columns, \
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
            for col in protected_attributes:
                # copy column (pandas warning)
                copy = sub_df[col].copy()
                # get unique values
                uniques = copy.unique().tolist()
                # remove current value
                uniques.remove(copy[index])
                # replace with random value in uniques
                copy[index] = uniques[np.random.randint(0, len(uniques) - 1)]
                # replace column
                sub_df[col] = copy

        # add mutant subset to dataframe
        df = pd.concat([df, sub_df])

    return df


########################################################################################################################
#                                                  New class operator                                                  #
########################################################################################################################
def new_class(dataframe, protected_attributes, mutation_ratio=0.0):
    assert type(dataframe) == pd.DataFrame, \
        'Type error: dataframe should be an instance of <pandas.core.frame.DataFrame>'
    assert isinstance(protected_attributes, list), 'Type error: protected_attributes should be an instance of list'
    for attribute in protected_attributes:
        assert isinstance(attribute, str), 'Type error: protected_attributes elements should be an instance of str'
        assert attribute in dataframe.columns, \
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

        for col in protected_attributes:
            sub_df = df.loc[random_index]
            if isinstance(sub_df[col][random_index[0]], str):
                df.loc[random_index, col] = 'null'

            else:
                df.loc[random_index, col] = max(df[col]) + 1

    return df
