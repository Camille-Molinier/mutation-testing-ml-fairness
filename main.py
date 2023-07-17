import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from fairpipes.utils import make_history_figs, make_multi_hist_dataframe
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

    N = 1000
    hist = multi_mutation_fairness_assessment(model, X_test, y_test, protected_attribute='sex',
                                              mutation_ratios=mutations, nb_iter=N)
    print()
    hist2 = multi_mutation_fairness_assessment(model, X_test, y_test, protected_attribute='sex',
                                               mutation_ratios=mutations, nb_iter=N)

    make_history_figs(hist, mutations)
    make_history_figs(hist2, mutations)

    results = make_multi_hist_dataframe(hist, mutations)
    results2 = make_multi_hist_dataframe(hist2, mutations)

    results.to_csv(f'./dat/exports/Adult_dt.csv')
    results2.to_csv(f'./dat/exports/Adult_dt_2.csv')

    for col in results.columns[1:]:
        _, p = mannwhitneyu(results[col], results2[col])
        print(f"{f'{col}' : <30} {p}")

    plt.show()


if __name__ == "__main__":
    main()
