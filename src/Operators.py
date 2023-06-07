from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


########################################################################################################################
#                                                      Interface                                                       #
########################################################################################################################
class Operator(ABC):
    @abstractmethod
    def compute_mutation(self):
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
