import pandas as pd
import numpy as np
from src.Operators import ColumnShuffle


def main():
    operator = ColumnShuffle(0.5)
    data = {
        'col1': np.random.randint(1, 100, size=100),
        'col2': np.random.rand(100),
        'col3': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], size=100)
    }

    # Crée le dataframe
    df = pd.DataFrame(data)
    df_mutated = operator.compute_mutation(df, ['col3'])
    print(sum(df['col3'] != df_mutated['col3'])/len(df))


if __name__ == "__main__":
    main()
