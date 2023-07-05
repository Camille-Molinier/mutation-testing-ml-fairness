import time
start = time.time()

import pandas as pd
import matplotlib.pyplot as plt

from fairpipes.utils import make_history_figs, hist_to_dataframe
from fairpipes.pipelines import multi_mutation_fairness_assessment

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split


def main():
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                    'income']
    df = pd.read_csv('./dat/datasets/adult.data', names=column_names)
    df = pd.DataFrame(OrdinalEncoder().fit_transform(df), columns=df.columns)

    trainset, testset = train_test_split(df, test_size=0.2, random_state=0)
    X_train, y_train = trainset.drop('income', axis=1), trainset.income
    X_test, y_test = testset.drop('income', axis=1), testset.income

    model = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)

    mutations = [0.1, 0.2, 0.3, 0.4, 0.5]

    hist = multi_mutation_fairness_assessment(model, X_test, y_test, protected_attribute='sex',
                                              mutation_ratios=mutations, nb_iter=500, output_name='Adult_dt')

    make_history_figs(hist, mutations)

    plt.show()
    print(f'total time: {time.time() - start}')


if __name__ == "__main__":
    main()
