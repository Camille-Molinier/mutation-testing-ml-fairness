import os
import time
from datetime import timedelta

import pandas as pd
from tqdm import tqdm

from fairpipes.pipelines import multi_mutation_fairness_assessment
from fairpipes.utils import make_history_figs, make_multi_hist_dataframe, make_multi_stats

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")


def main():
    # load data
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                    'income']
    data = pd.read_csv('./dat/datasets/adult.data', names=column_names)
    df = pd.DataFrame(OrdinalEncoder().fit_transform(data), columns=data.columns)

    # split into train and test set
    trainset, testset = train_test_split(df, test_size=0.2)

    X_train, y_train = trainset.drop('income', axis=1), trainset.income
    X_test, y_test = testset.drop('income', axis=1), testset.income

    # initialize models
    # dt = DecisionTreeClassifier()
    # dt_opt = DecisionTreeClassifier(max_depth=1, ccp_alpha=1e-10, min_samples_leaf=1)
    #
    # rf = RandomForestClassifier(n_jobs=100)
    # rf_opt = RandomForestClassifier(max_depth=1, n_estimators=44, bootstrap=True, oob_score=True, n_jobs=100)
    #
    # xgbc = XGBClassifier(n_jobs=100)
    # xgbc_opt = XGBClassifier(max_depth=5, booster='gbtree', n_estimators=1, learning_rate=1e-7, n_jobs=100)
    #
    # svm = SVC()
    svm_opt = SVC(kernel='rbf', C=1.777777777777, gamma=3.1622776601683795e-05)
    #
    # ada = AdaBoostClassifier()
    
    # ada_opt = AdaBoostClassifier(estimator=RandomForestClassifier(), learning_rate=1e-10)
    #
    # knn = KNeighborsClassifier()
    # knn_opt = KNeighborsClassifier(n_neighbors=7, p=1, leaf_size=1)

    # sklean_MLPC = MLPClassifier()
    # sklean_MLPC_opt = MLPClassifier(solver='adam', max_iter=500, hidden_layer_sizes=(1024, 1024))

    # pipeline configuration
    models = [
        # dt,
        # dt_opt,
        # rf,
        # rf_opt,
        # xgbc,
        # xgbc_opt,
        # ada,
        # ada_opt,
        # knn,
        # knn_opt,
        # svm,
        svm_opt,
        # sklean_MLPC,
        # sklean_MLPC_opt
    ]

    names = [
        # 'Decision_tree',
        # 'Decision_tree_optimized',
        # 'Random_forest',
        # 'Random_forest_optimized',
        # 'XGBC',
        # 'XGBC_optimized',
        # 'AdaBoost',
        # 'AdaBoost_optimized',
        # 'KNN',
        # 'KNN_optimized',
        # 'svm',
        'svm_opt',
        # 'sklean_MLPC',
        # 'sklean_MLPC_opt'
    ]

    mutations = [0.1, 0.2, 0.3, 0.4, 0.5]
    dataset = 'adult'
    protected_attribute = 'sex'
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
        fig_save_path = f'./dat/fig/{dataset}/{protected_attribute}/{names[i]}'
        raw_save_path = f'./dat/exports/{dataset}/{protected_attribute}'

        if not os.path.exists(fig_save_path):
            os.makedirs(fig_save_path)

        if not os.path.exists(raw_save_path):
            os.makedirs(raw_save_path)

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
                          save_path=f'{fig_save_path}/history.png',
                          display=False)

        # compute p-values and effect sizes
        make_multi_stats(hist,
                         mutations,
                         model_name=f'{names[i]}',
                         p_val_save_path=f'{fig_save_path}/p-values.png',
                         effect_size_save_path=f'{fig_save_path}/effect_sizes.png',
                         display=False)

        # save raw history in csv format
        csv = make_multi_hist_dataframe(hist, mutations)
        csv.to_csv(f'{raw_save_path}/{names[i]}.csv')

        print()


if __name__ == "__main__":
    start = time.time()
    main()
    print(f'Total time : {timedelta(seconds=time.time() - start)}')
