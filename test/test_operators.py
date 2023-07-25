import unittest
import numpy as np
import pandas as pd
from fairpipes.operators import column_shuffle, column_killing, redistribution, new_class


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

    def test_default_parameters(self):
        try:
            column_shuffle(self.df, 'col3')
        except Exception:
            self.fail("column_shuffle() raised Exception unexpectedly!")

    def test_not_dataframe_parameter(self):
        with self.assertRaises(AssertionError):
            column_shuffle('loutre', 'col3')

    def test_not_string_parameter(self):
        with self.assertRaises(AssertionError):
            column_shuffle(self.df, self.df)

    def test_not_float_shuffle_ratio(self):
        with self.assertRaises(AssertionError):
            column_shuffle(self.df, self.df, 'loutre')

    def test_not_float_shuffle_ratio2(self):
        with self.assertRaises(AssertionError):
            column_shuffle(self.df, self.df, self.df)

    def test_not_float_shuffle_ratio3(self):
        with self.assertRaises(AssertionError):
            column_shuffle(self.df, self.df, 1)

    def test_incorrect_shuffle_ratio_value(self):
        with self.assertRaises(AssertionError):
            column_shuffle(self.df, self.df, 1.5)

    def test_incorrect_shuffle_ratio_value2(self):
        with self.assertRaises(AssertionError):
            column_shuffle(self.df, self.df, -1.5)

    def test_list_elements_are_in_dataframe_columns(self):
        with self.assertRaises(AssertionError):
            column_shuffle(self.df, 'col4')

    def test_computation_doing_something(self):
        shuffled = column_shuffle(self.df, 'col3', 0.5)
        self.assertFalse(self.df['col3'].equals(shuffled['col3']))


########################################################################################################################
#                                               Column dropping operator                                               #
########################################################################################################################
class ColumnKillingOperatorTest(unittest.TestCase):

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
            column_killing('loutre', 'col3')

    def test_not_string_parameter(self):
        with self.assertRaises(AssertionError):
            column_killing(self.df, self.df)

    def test_column_is_in_dataframe_columns(self):
        with self.assertRaises(AssertionError):
            column_killing(self.df, 'col4')

    def test_computation_do_something(self):
        mutant = column_killing(self.df, 'col3')
        self.assertFalse(self.df['col3'].equals(mutant['col3']))

    def test_all_vales_are_the_same(self):
        mutant = column_killing(self.df, 'col3')
        self.assertTrue((mutant['col3'] == mutant['col3'][0]).all())


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
            redistribution('loutre', 'col3')

    def test_not_list_of_string_parameter(self):
        with self.assertRaises(AssertionError):
            redistribution(self.df, self.df)

    def test_column_is_in_dataframe_columns(self):
        with self.assertRaises(AssertionError):
            redistribution(self.df, 'col4')

    def test_computation_do_something(self):
        shuffled = redistribution(self.df, 'col1')
        self.assertFalse(self.df['col1'].equals(shuffled['col1']))


########################################################################################################################
#                                                  New class operator                                                  #
########################################################################################################################
class NewClassTest(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(1)
        data = {
            'col1': np.random.randint(1, 20, size=100),
            'col2': np.random.rand(100),
            'col3': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], size=100)
        }

        # Crée le dataframe
        self.df = pd.DataFrame(data)

    def test_default_parameters(self):
        try:
            new_class(self.df, 'col3')
        except Exception:
            self.fail("duplication_mutation() raised Exception unexpectedly!")

    def test_not_dataframe_parameter(self):
        with self.assertRaises(AssertionError):
            new_class('loutre', 'col3')

    def test_not_string_parameter(self):
        with self.assertRaises(AssertionError):
            new_class(self.df, self.df)

    def test_not_float_shuffle_ratio2(self):
        with self.assertRaises(AssertionError):
            new_class(self.df, self.df, self.df)

    def test_not_float_shuffle_ratio3(self):
        with self.assertRaises(AssertionError):
            new_class(self.df, self.df, 1)

    def test_incorrect_shuffle_ratio_value(self):
        with self.assertRaises(AssertionError):
            new_class(self.df, self.df, 1.5)

    def test_incorrect_shuffle_ratio_value2(self):
        with self.assertRaises(AssertionError):
            new_class(self.df, self.df, -1.5)

    def test_computation_do_something_with_int(self):
        shuffled = new_class(self.df, 'col1', 0.5)
        print(max(self.df['col1']), " ", max(shuffled['col1']))
        self.assertFalse(self.df['col1'].equals(shuffled['col1']))

    def test_computation_do_something_with_float(self):
        shuffled = new_class(self.df, 'col2', 0.5)
        print(max(self.df['col2']), " ", max(shuffled['col2']))
        self.assertFalse(self.df['col2'].equals(shuffled['col2']))

    def test_computation_do_something_with_string(self):
        shuffled = new_class(self.df, 'col3', 0.5)
        print(max(self.df['col3']), " ", max(shuffled['col3']))
        self.assertFalse(self.df['col3'].equals(shuffled['col3']))


def make_suite():
    suite = unittest.TestSuite()
    suite.addTest(ColumnShuffleOperatorTest())
    suite.addTest(ColumnKillingOperatorTest())
    suite.addTest(RedistributionOperatorTest())
    suite.addTest(NewClassTest())
    return suite


def run_operator_tests():
    runner = unittest.TextTestRunner()
    runner.run(make_suite())


if __name__ == '__main__':
    run_operator_tests()
