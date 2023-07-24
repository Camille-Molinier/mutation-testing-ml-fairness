import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from fairpipes.pipelines import multi_mutation_fairness_assessment
from fairpipes.utils import make_history_figs, make_multi_hist_dataframe, make_multi_stats

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split


def main():
    # load data
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                    'income']
    df = pd.read_csv('./dat/datasets/adult.data', names=column_names)
    df = pd.DataFrame(OrdinalEncoder().fit_transform(df), columns=df.columns)

    # split into train and test set
    trainset, testset = train_test_split(df, test_size=0.2, random_state=0)
    X_train, y_train = trainset.drop('income', axis=1), trainset.income
    X_test, y_test = testset.drop('income', axis=1), testset.income

    # initialize models
    dt = DecisionTreeClassifier(random_state=0)
    dt_opt = DecisionTreeClassifier(random_state=0, max_depth=14, ccp_alpha=0.0003020408163265306,
                                    min_samples_leaf=13)

    rf = RandomForestClassifier(random_state=0, n_jobs=100)
    rf_opt = RandomForestClassifier(random_state=0, max_depth=50, n_estimators=538, bootstrap=True,
                                    oob_score=True, n_jobs=100)

    xgbc = XGBClassifier(random_state=0, n_jobs=100)
    xgbc_opt = XGBClassifier(random_state=0, max_depth=6, n_estimators=180, learning_rate=0.04777777777777778,
                             n_jobs=100)

    # svm = SVC()
    # svm_opt = SVC(C=1.777777777777, gamma=3.1622776601683795e-05)

    ada = AdaBoostClassifier(random_state=0)
    ada_opt = AdaBoostClassifier(random_state=0, estimator=RandomForestClassifier(), n_estimators=4, learning_rate=1e-7)

    knn = KNeighborsClassifier()
    knn_opt = KNeighborsClassifier(n_neighbors=6, algorithm='kd_tree', p=1)

    # pipeline configuration
    models = [dt, dt_opt, rf, rf_opt, xgbc, xgbc_opt, ada, ada_opt, knn, knn_opt]
    names = ['Decision_tree',
             'Decision_tree_optimized',
             'Random_forest',
             'Random_forest_optimized',
             'XGBC',
             'XGBC_optimized',
             'AdaBoost',
             'AdaBoost_optimized',
             'KNN',
             'KNN_optimized']
    mutations = [0.1, 0.2, 0.3, 0.4, 0.5]
    protected_attribute = 'marital-status'
    nb_iter = 50

    # train models
    print('Training models :')
    for model in tqdm(models):
        model.fit(X_train, y_train)
    print()

    # Pipeline
    for i, model in enumerate(models):
        print(f'{names[i]} assessment :')
        # check if save path exist, otherwise create it
        save_path = f'./dat/fig/adult/{protected_attribute}/{names[i]}'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # pipeline call
        hist = multi_mutation_fairness_assessment(model,
                                                  X_test,
                                                  y_test,
                                                  protected_attribute=protected_attribute,
                                                  mutation_ratios=mutations,
                                                  nb_iter=nb_iter)

        # make recap fig
        make_history_figs(hist,
                          mutations,
                          title=f'Adult {names[i]} operators response',
                          save_path=f'./dat/fig/adult/{protected_attribute}/{names[i]}/history.png',
                          display=False)

        # compute p-values and effect sizes
        make_multi_stats(hist,
                         mutations,
                         model_name=f'{names[i]}',
                         p_val_save_path=f'./dat/fig/adult/{protected_attribute}/{names[i]}/p-values.png',
                         effect_size_save_path=f'./dat/fig/adult/{protected_attribute}/{names[i]}/effect_sizes.png',
                         display=False)

        # save raw history in csv format
        csv = make_multi_hist_dataframe(hist, mutations)
        csv.to_csv(f'./dat/exports/adult/{protected_attribute}/{names[i]}.csv')

        print()


if __name__ == "__main__":
    main()
