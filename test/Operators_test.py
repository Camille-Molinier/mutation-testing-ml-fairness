import unittest
import numpy as np
import pandas as pd
from src.Operators import ColumnShuffle


########################################################################################################################
#                                               Column shuffle operator                                                #
########################################################################################################################
class ColumnShuffleOperatorTest(unittest.TestCase):

    def setUp(self) -> None:
        self.operator = ColumnShuffle(0.5)
        np.random.seed(1)  # Pour la reproductibilité
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

    def test_computatation_doing_something(self):
        shuffled = self.operator.compute_mutation(self.df, ['col3'])
        self.assertFalse(self.df['col3'].equals(shuffled['col3']))

    def test_mutation_proportion(self):
        for i in range(1000):
            shuffled = self.operator.compute_mutation(self.df, ['col3'])
            ratio = sum(self.df['col3'] != shuffled['col3']) / len(self.df)
            self.assertTrue(self.operator.shuffle_ratio >= ratio >= self.operator.shuffle_ratio - 0.5)


def make_suite():
    suite = unittest.TestSuite()
    suite.addTest(ColumnShuffleOperatorTest())
    return suite


def run_operator_tests():
    runner = unittest.TextTestRunner()
    runner.run(make_suite())


if __name__ == '__main__':
    run_operator_tests()
