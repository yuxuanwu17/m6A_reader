from one_hot import seq_to_mat
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path, engine='python', error_bad_lines=False)
    train_All_1 = df.iloc[:, 2]
    test_all_1 = df.iloc[:, 3]

    X_train = np.array(train_All_1)
    lt = []
    for seq in X_train:
        x = seq_to_mat(seq)
        lt.append(x)
    x_train = np.array(lt)

    test = DataFrame(test_all_1)
    test = test.dropna()
    lst_test = []
    x_val = test_all_1[0:test.shape[0], ]

    for seqs in x_val:
        x = seq_to_mat(seqs)
        lst_test.append(x)

    x_val = np.array(lst_test)

    y_train = np.array([1, 0])
    y_train = y_train.repeat(train_All_1.shape[0] / 2)
    y_train = np.mat(y_train).transpose()

    y_val = np.array([1, 0])
    y_val = y_val.repeat(test.shape[0] / 2)
    y_val = np.mat(y_val).transpose()

    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5)

    return x_train, x_test, x_val, y_test, y_train, y_val
