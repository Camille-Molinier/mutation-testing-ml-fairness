import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, confusion_matrix

import fairpipes.operators
from fairpipes.pipelines import basic_fairness_assessment
import tensorflow as tf
from fairpipes.models import TensorflowModel
import numpy as np
from tqdm import tqdm


def main():
    df = pd.read_csv('datasets/bank.csv', sep=';')
    df = pd.DataFrame(OrdinalEncoder().fit_transform(df), columns=df.columns)

    trainset, testset = train_test_split(df, test_size=0.3)
    trainset, testset = trainset.reset_index(), testset.reset_index()
    X_train, y_train = trainset.drop("y", axis=1), trainset.y
    X_test, y_test = testset.drop("y", axis=1), testset.y

    batch_size = 64
    train_data = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values)).batch(batch_size)

    train_data = train_data.shuffle(buffer_size=len(X_train))

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(1, activation='sigmoid')
    # ])
    #
    # model.compile(loss='binary_crossentropy', optimizer='adam')
    #
    # model.fit(train_data, epochs=10)
    #
    # Model = TensorflowModel(model, y_test.unique())
    # class_names = np.unique(y_test)
    # predictions = Model.predict([X_test])[0]

    models = [DecisionTreeClassifier(), RandomForestClassifier(), SVC(), AdaBoostClassifier()]

    for model in models:
        model.fit(X_train, y_train)

    models.append(model)

    model_ = DecisionTreeClassifier()
    model_.fit(X_train, y_train)

    # start = time.time()
    # fairpipes.operators.column_shuffle(X_test, 'marital', shuffle_ratio=0.3, tol=0.2)
    # print(f'redistribution: {time.time()-start}')
    #
    # start = time.time()
    # fairpipes.operators.redistribution(X_test, 'marital')
    # print(f'redistribution: {time.time()-start}')

    mutations = [0.1, 0.2, 0.3, 0.4, 0.5]
    tol = [0.1, 0.15, 0.2, 0.25, 0.25]
    for i in range(len(mutations)):
        res, history = basic_fairness_assessment(model_, X_test, y_test, ['marital'], mutation_ratio=mutations[i],
                                                 tol=tol[i], nb_iter=5000)
        make_fing(history)
    plt.show()


def make_fing(history):
    names = list(history.keys())
    accuracies = [history[key]['accuracy'] for key in names]
    dpd = [history[key]['dpd'] for key in names]
    eod = [history[key]['eod'] for key in names]
    fnr = [history[key]['fnr'] for key in names]
    tnr = [history[key]['tnr'] for key in names]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    axes[1][2].set_visible(False)

    axes[1][0].set_position([0.24, 0.125, 0.228, 0.343])
    axes[1][1].set_position([0.55, 0.125, 0.228, 0.343])

    sns.violinplot(data=accuracies, palette='turbo', inner=None, linewidth=0, saturation=0.4, ax=axes[0][0])
    sns.boxplot(data=accuracies, palette='turbo', width=0.3, boxprops={'zorder': 2}, ax=axes[0][0])
    axes[0][0].set_xticks(range(len(names)), names)
    axes[0][0].set_xticklabels(names, rotation=30)
    axes[0][0].set_ylabel('Accuracy')

    sns.violinplot(data=dpd, palette='turbo', inner=None, linewidth=0, saturation=0.4, ax=axes[0][1])
    sns.boxplot(data=dpd, palette='turbo', width=0.3, boxprops={'zorder': 2}, ax=axes[0][1])
    axes[0][1].set_xticks(range(len(names)), names)
    axes[0][1].set_xticklabels(names, rotation=30)
    axes[0][1].set_ylabel('dpd')

    sns.violinplot(data=eod, palette='turbo', inner=None, linewidth=0, saturation=0.4, ax=axes[0][2])
    sns.boxplot(data=eod, palette='turbo', width=0.3, boxprops={'zorder': 2}, ax=axes[0][2])
    axes[0][2].set_xticks(range(len(names)), names)
    axes[0][2].set_xticklabels(names, rotation=30)
    axes[0][2].set_ylabel('eod')

    sns.violinplot(data=fnr, palette='turbo', inner=None, linewidth=0, saturation=0.4, ax=axes[1][0])
    sns.boxplot(data=fnr, palette='turbo', width=0.3, boxprops={'zorder': 2}, ax=axes[1][0])
    axes[1][0].set_xticks(range(len(names)), names)
    axes[1][0].set_xticklabels(names, rotation=30)
    axes[1][0].set_ylabel('fnr')

    sns.violinplot(data=tnr, palette='turbo', inner=None, linewidth=0, saturation=0.4, ax=axes[1][1])
    sns.boxplot(data=tnr, palette='turbo', width=0.3, boxprops={'zorder': 2}, ax=axes[1][1])
    axes[1][1].set_xticks(range(len(names)), names)
    axes[1][1].set_xticklabels(names, rotation=30)
    axes[1][1].set_ylabel('tnr')


if __name__ == "__main__":
    main()
