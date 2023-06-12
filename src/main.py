import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from src.Operators import column_shuffle, column_dropping, redistribution, duplication_mutation, new_class
from fairlearn.metrics import demographic_parity_ratio
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_csv('../datasets/bank.csv', sep=';')
    df = pd.DataFrame(OrdinalEncoder().fit_transform(df), columns=df.columns)

    df = redistribution(df, ['marital'])
    trainset, testset = train_test_split(df, test_size=0.3)
    X_train, y_train = trainset.drop("y", axis=1), trainset.y
    X_test, y_test = testset.drop("y", axis=1), testset.y
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(demographic_parity_ratio(y_test, y_pred, sensitive_features=testset.marital))


if __name__ == "__main__":
    main()
