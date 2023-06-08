import unittest
import numpy as np
import pandas as pd
from src.Operators import ColumnShuffle, ColumnDropping


########################################################################################################################
#                                               Column shuffle operator                                                #
########################################################################################################################
class ColumnShuffleOperatorTest(unittest.TestCase):

    def setUp(self) -> None:
        self.operator = ColumnShuffle(0.5)
        np.random.seed(1)
        data = {
            'col1': np.random.randint(1, 100, size=100),
            'col2': np.random.rand(100),
            'col3': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], size=100)
        }

        # Crée le dataframe
        self.df = pd.DataFrame(data)

    def test_constructor(self):
        self.assertIsNotNone(ColumnShuffle(0.5))

    def test_empty_constructor(self):
        self.assertIsNotNone(ColumnShuffle())

    def test_correct_shuffle_value(self):
        self.assertEqual(0.5, self.operator.shuffle_ratio)

    def test_correct_shuffle_value_with_empty_constructor(self):
        self.assertEqual(0, ColumnShuffle().shuffle_ratio)

    def test_not_dataframe_parameter(self):
        with self.assertRaises(AssertionError):
            self.operator.compute_mutation('loutre', ['col3'])

    def test_not_list_of_string_parameter(self):
        with self.assertRaises(AssertionError):
            self.operator.compute_mutation(self.df, self.df)

    def test_list_of_non_str_elements(self):
        with self.assertRaises(AssertionError):
            self.operator.compute_mutation(self.df, [self.df])

    def test_list_of_mixed_str_elements(self):
        with self.assertRaises(AssertionError):
            self.operator.compute_mutation(self.df, [self.df])

    def test_list_of_mixed_str_elements2(self):
        with self.assertRaises(AssertionError):
            self.operator.compute_mutation(self.df, ['col2', 'otter'])

    def test_list_of_mixed_str_elements3(self):
        with self.assertRaises(AssertionError):
            self.operator.compute_mutation(self.df, [0.5])

    def test_list_elements_are_in_dataframe_columns(self):
        with self.assertRaises(AssertionError):
            self.operator.compute_mutation(self.df, ['col4'])

    def test_computation_doing_something(self):
        shuffled = self.operator.compute_mutation(self.df, ['col3'])
        self.assertFalse(self.df['col3'].equals(shuffled['col3']))

    def test_mutation_proportion(self):
        for i in range(1000):
            shuffled = self.operator.compute_mutation(self.df, ['col3'])
            ratio = sum(self.df['col3'] != shuffled['col3']) / len(self.df)
            self.assertTrue(self.operator.shuffle_ratio >= ratio >= self.operator.shuffle_ratio - 0.5)


########################################################################################################################
#                                               Column dropping operator                                               #
########################################################################################################################
class ColumnDroppingOperatorTest(unittest.TestCase):

    def setUp(self) -> None:
        self.operator = ColumnDropping()
        np.random.seed(1)
        data = {
            'col1': np.random.randint(1, 100, size=100),
            'col2': np.random.rand(100),
            'col3': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], size=100)
        }

        # Crée le dataframe
        self.df = pd.DataFrame(data)

    def test_constructor(self):
        self.assertIsNotNone(ColumnShuffle())

    def test_not_empty_constructor(self):
        with self.assertRaises(TypeError):
            ColumnDropping(0.5)

    def test_not_dataframe_parameter(self):
        with self.assertRaises(AssertionError):
            self.operator.compute_mutation('loutre', ['col3'])

    def test_not_list_of_string_parameter(self):
        with self.assertRaises(AssertionError):
            self.operator.compute_mutation(self.df, self.df)

    def test_list_of_non_str_elements(self):
        with self.assertRaises(AssertionError):
            self.operator.compute_mutation(self.df, [self.df])

    def test_list_of_mixed_str_elements(self):
        with self.assertRaises(AssertionError):
            self.operator.compute_mutation(self.df, [self.df])

    def test_list_of_mixed_str_elements2(self):
        with self.assertRaises(AssertionError):
            self.operator.compute_mutation(self.df, ['col2', 'otter'])

    def test_list_of_mixed_str_elements3(self):
        with self.assertRaises(AssertionError):
            self.operator.compute_mutation(self.df, [0.5])

    def test_list_elements_are_in_dataframe_columns(self):
        with self.assertRaises(AssertionError):
            self.operator.compute_mutation(self.df, ['col4'])

    def test_computation_do_something(self):
        mutant = self.operator.compute_mutation(self.df, ['col3'])
        self.assertEqual(len(self.df.columns)-1, len(mutant.columns))

    def test_drop_all_columns(self):
        mutant = self.operator.compute_mutation(self.df, ['col1', 'col2', 'col3'])
        self.assertEqual(0, len(mutant.columns))


########################################################################################################################
#                                               Redistribution operator                                                #
########################################################################################################################
class ColumnDroppingOperatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.operator = ColumnDropping()
        np.random.seed(1)
        data = {
            'col1': np.random.randint(1, 100, size=100),
            'col2': np.random.rand(100),
            'col3': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], size=100)
        }

        # Crée le dataframe
        self.df = pd.DataFrame(data)

    def test_constructor(self):
        self.assertIsNotNone(ColumnShuffle())

    def test_not_empty_constructor(self):
        with self.assertRaises(TypeError):
            ColumnDropping(0.5)

    def test_not_dataframe_parameter(self):
        with self.assertRaises(AssertionError):
            self.operator.compute_mutation('loutre', ['col3'])

    def test_not_list_of_string_parameter(self):
        with self.assertRaises(AssertionError):
            self.operator.compute_mutation(self.df, self.df)

    def test_list_of_non_str_elements(self):
        with self.assertRaises(AssertionError):
            self.operator.compute_mutation(self.df, [self.df])

    def test_list_of_mixed_str_elements(self):
        with self.assertRaises(AssertionError):
            self.operator.compute_mutation(self.df, [self.df])

    def test_list_of_mixed_str_elements2(self):
        with self.assertRaises(AssertionError):
            self.operator.compute_mutation(self.df, ['col2', 'otter'])

    def test_list_of_mixed_str_elements3(self):
        with self.assertRaises(AssertionError):
            self.operator.compute_mutation(self.df, [0.5])

    def test_list_elements_are_in_dataframe_columns(self):
        with self.assertRaises(AssertionError):
            self.operator.compute_mutation(self.df, ['col4'])

    def test_computation_do_something(self):
        mutant = self.operator.compute_mutation(self.df, ['col3'])
        self.assertEqual(len(self.df.columns)-1, len(mutant.columns))

    def test_drop_all_columns(self):
        mutant = self.operator.compute_mutation(self.df, ['col1', 'col2', 'col3'])
        self.assertEqual(0, len(mutant.columns))


def make_suite():
    suite = unittest.TestSuite()
    suite.addTest(ColumnShuffleOperatorTest())
    suite.addTest(ColumnDroppingOperatorTest())
    return suite


def run_operator_tests():
    runner = unittest.TextTestRunner()
    runner.run(make_suite())


if __name__ == '__main__':
    run_operator_tests()
