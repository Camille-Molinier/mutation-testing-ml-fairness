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

    def compute_mutation(self, dataframe, protected_attribute):
        assert type(dataframe) == pd.DataFrame, \
            'Type error: dataframe should be an instance of <pandas.core.frame.DataFrame>'

        # copy dataframe to manipulate it safely
        df = dataframe.copy()

        # run through protected attributes columns
        for col in protected_attribute:
            # compute number of row to shuffle
            n_samples = int(len(df) * self.shuffle_ratio)
            # choose n_samples random indexes
            random_index = df.sample(n=n_samples).index.tolist()

            # loop to reduce shuffle vanishing
            ratio = 0
            while ratio <= self.shuffle_ratio - 0.05:
                # make a permutation of rows
                shuffled = np.random.permutation(df.loc[random_index, col])
                # replace in data
                df.loc[random_index, 'col3'] = shuffled
                # compute shuffle ratio
                ratio = sum(df['col3'] != dataframe['col3']) / len(dataframe)

        return df
