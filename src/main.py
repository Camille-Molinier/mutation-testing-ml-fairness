import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.Operators import column_shuffle, column_dropping, redistribution


def main():
    N_samples = 100000
    data = {
        'col1': np.random.randint(1, 100, size=N_samples),
        'col2': np.random.rand(N_samples),
        'col3': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], size=N_samples)
    }

    # Crée le dataframe
    df = pd.DataFrame(data)
    df_mutated = redistribution(df, ['col2'])

    print(type(df_mutated['col2'][0]))
    plt.figure()
    df['col2'].value_counts().plot.pie()
    plt.figure()
    df_mutated['col2'].value_counts().plot.pie()
    plt.show()


if __name__ == "__main__":
    main()
