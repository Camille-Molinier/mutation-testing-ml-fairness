import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.Operators import ColumnShuffle, ColumnDropping, Redistribution


def main():
    operator = Redistribution()
    N_samples = 101
    data = {
        'col1': np.random.randint(1, 100, size=N_samples),
        'col2': np.random.rand(N_samples),
        'col3': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], size=N_samples)
    }

    # Crée le dataframe
    df = pd.read_csv('../datasets/compas-scores-two-years.csv')
    df_mutated = operator.compute_mutation(df, ['race'])

    plt.figure()
    df['race'].value_counts().plot.pie()
    plt.figure()
    df_mutated['race'].value_counts().plot.pie()
    plt.show()


if __name__ == "__main__":
    main()
