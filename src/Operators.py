from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import random


########################################################################################################################
#                                                      Interface                                                       #
########################################################################################################################
class Operator(ABC):
    @abstractmethod
    def compute_mutation(self, dataframe, protected_attributes):
        pass


########################################################################################################################
#                                               Column shuffle operator                                                #
########################################################################################################################
class ColumnShuffle(Operator):
    """
    Dataset mutation operator
    shuffle protected attributes values to create noise
    """

    def __init__(self, shuffle_ratio=0):
        self.shuffle_ratio = shuffle_ratio

    def compute_mutation(self, dataframe, protected_attributes):
        assert type(dataframe) == pd.DataFrame, \
            'Type error: dataframe should be an instance of <pandas.core.frame.DataFrame>'
        assert isinstance(protected_attributes, list), 'Type error: protected_attributes should be an instance of list'
        for attribute in protected_attributes:
            assert isinstance(attribute, str), 'Type error: protected_attributes elements should be an instance of str'
            assert attribute in dataframe.columns, \
                'Key error: protected_attribute elements should be in dataframe columns'

        # copy dataframe to manipulate it safely
        df = dataframe.copy()

        # run through protected attributes columns
        for attribute in protected_attributes:
            # compute number of row to shuffle
            n_samples = int(len(df) * self.shuffle_ratio)
            # choose n_samples random indexes
            random_index = df.sample(n=n_samples).index.tolist()

            # loop to reduce shuffle vanishing
            ratio = 0
            while ratio <= self.shuffle_ratio - 0.05:
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
class ColumnDropping(Operator):
    def compute_mutation(self, dataframe, protected_attributes):
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
class Redistribution(Operator):
    def compute_mutation(self, dataframe, protected_attributes):
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
            # compute distribution
            distribution = df_col.value_counts()

            # compute balanced repartition
            redistribution_factor = int(len(df_col)/len(distribution))
            redistribution_leftover = len(df_col) - (redistribution_factor*len(distribution))
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
            rest = optimal_distribution[optimal_distribution>0]
            rest_list = [str(index) for index, value in rest.items() for e in range(value)]
            random.shuffle(rest_list)

            # place remaining value in unchanged indexes
            copy = df_col.copy()
            copy.loc[unchanged] = rest_list

            # replace dataframe column with balanced version
            df[col] = copy

        return df
