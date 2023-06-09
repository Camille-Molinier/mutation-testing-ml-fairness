import unittest
import numpy as np
import pandas as pd
from src.Operators import column_shuffle, column_dropping, redistribution


########################################################################################################################
#                                               Column shuffle operator                                                #
########################################################################################################################
class ColumnShuffleOperatorTest(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(1)
        data = {
            'col1': np.random.randint(1, 100, size=100),
            'col2': np.random.rand(100),
            'col3': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], size=100)
        }

        # Crée le dataframe
        self.df = pd.DataFrame(data)

    def test_not_dataframe_parameter(self):
        with self.assertRaises(AssertionError):
            column_shuffle('loutre', ['col3'])

    def test_not_list_of_string_parameter(self):
        with self.assertRaises(AssertionError):
            column_shuffle(self.df, self.df)

    def test_list_of_non_str_elements(self):
        with self.assertRaises(AssertionError):
            column_shuffle(self.df, [self.df])

    def test_list_of_mixed_str_elements(self):
        with self.assertRaises(AssertionError):
            column_shuffle(self.df, [self.df])

    def test_list_of_mixed_str_elements2(self):
        with self.assertRaises(AssertionError):
            column_shuffle(self.df, ['col2', 'otter'])

    def test_list_of_mixed_str_elements3(self):
        with self.assertRaises(AssertionError):
            column_shuffle(self.df, [0.5])

    def test_list_elements_are_in_dataframe_columns(self):
        with self.assertRaises(AssertionError):
            column_shuffle(self.df, ['col4'])

    def test_computation_doing_something(self):
        shuffled = column_shuffle(self.df, ['col3'], 0.5)
        self.assertFalse(self.df['col3'].equals(shuffled['col3']))

    def test_mutation_proportion(self):
        shuffle_ratio = 0.5
        tol = 0.05
        for i in range(1000):
            shuffled = column_shuffle(self.df, ['col3'], shuffle_ratio, tol)
            ratio = sum(self.df['col3'] != shuffled['col3']) / len(self.df)
            self.assertTrue(shuffle_ratio >= ratio >= shuffle_ratio - tol)


########################################################################################################################
#                                               Column dropping operator                                               #
########################################################################################################################
class ColumnDroppingOperatorTest(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(1)
        data = {
            'col1': np.random.randint(1, 100, size=100),
            'col2': np.random.rand(100),
            'col3': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], size=100)
        }

        # Crée le dataframe
        self.df = pd.DataFrame(data)

    def test_not_dataframe_parameter(self):
        with self.assertRaises(AssertionError):
            column_dropping('loutre', ['col3'])

    def test_not_list_of_string_parameter(self):
        with self.assertRaises(AssertionError):
            column_dropping(self.df, self.df)

    def test_list_of_non_str_elements(self):
        with self.assertRaises(AssertionError):
            column_dropping(self.df, [self.df])

    def test_list_of_mixed_str_elements(self):
        with self.assertRaises(AssertionError):
            column_dropping(self.df, [self.df])

    def test_list_of_mixed_str_elements2(self):
        with self.assertRaises(AssertionError):
            column_dropping(self.df, ['col2', 'otter'])

    def test_list_of_mixed_str_elements3(self):
        with self.assertRaises(AssertionError):
            column_dropping(self.df, [0.5])

    def test_list_elements_are_in_dataframe_columns(self):
        with self.assertRaises(AssertionError):
            column_dropping(self.df, ['col4'])

    def test_computation_do_something(self):
        mutant = column_dropping(self.df, ['col3'])
        self.assertEqual(len(self.df.columns) - 1, len(mutant.columns))

    def test_drop_all_columns(self):
        mutant = column_dropping(self.df, ['col1', 'col2', 'col3'])
        self.assertEqual(0, len(mutant.columns))


########################################################################################################################
#                                               Redistribution operator                                                #
########################################################################################################################
class RedistributionOperatorTest(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(1)
        data = {
            'col1': np.random.randint(1, 20, size=100),
            'col2': np.random.rand(100),
            'col3': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], size=100)
        }

        # Crée le dataframe
        self.df = pd.DataFrame(data)

    def test_not_dataframe_parameter(self):
        with self.assertRaises(AssertionError):
            redistribution('loutre', ['col3'])

    def test_not_list_of_string_parameter(self):
        with self.assertRaises(AssertionError):
            redistribution(self.df, self.df)

    def test_list_of_non_str_elements(self):
        with self.assertRaises(AssertionError):
            redistribution(self.df, [self.df])

    def test_list_of_mixed_str_elements(self):
        with self.assertRaises(AssertionError):
            redistribution(self.df, [self.df])

    def test_list_of_mixed_str_elements2(self):
        with self.assertRaises(AssertionError):
            redistribution(self.df, ['col2', 'otter'])

    def test_list_of_mixed_str_elements3(self):
        with self.assertRaises(AssertionError):
            redistribution(self.df, [0.5])

    def test_list_elements_are_in_dataframe_columns(self):
        with self.assertRaises(AssertionError):
            redistribution(self.df, ['col4'])

    def test_computation_do_something(self):
        shuffled = redistribution(self.df, ['col1'])
        self.assertFalse(self.df['col1'].equals(shuffled['col1']))


def make_suite():
    suite = unittest.TestSuite()
    suite.addTest(ColumnShuffleOperatorTest())
    suite.addTest(ColumnDroppingOperatorTest())
    suite.addTest(RedistributionOperatorTest())
    return suite


def run_operator_tests():
    runner = unittest.TextTestRunner()
    runner.run(make_suite())


if __name__ == '__main__':
    run_operator_tests()
