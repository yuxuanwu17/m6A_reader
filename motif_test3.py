import matplotlib
from sklearn.model_selection import train_test_split

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tensorflow as tf
import numpy as np
# from keras import Sequential, Model
from tensorflow.python.keras.models import Sequential, Model
import pandas as pd
from deepexplain.tensorflow import DeepExplain
from tensorflow.python.keras.models import load_model
import os
from pandas import DataFrame
import keras
from keras import backend as K

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


def seq_to_mat(seq):
    seq_len = len(seq)
    seq = seq.replace('A', '0')
    seq = seq.replace('a', '0')
    seq = seq.replace('C', '1')
    seq = seq.replace('c', '1')
    seq = seq.replace('G', '2')
    seq = seq.replace('g', '2')
    seq = seq.replace('T', '3')
    seq = seq.replace('t', '3')
    seq = seq.replace('U', '3')
    seq = seq.replace('u', '3')
    seq = seq.replace('N', '4')
    seq = seq.replace('n', '4')
    seq_code = np.zeros((4, seq_len), dtype='float16')
    for i in range(seq_len):
        if int(seq[i]) != 4:
            seq_code[int(seq[i]), i] = 1
        else:
            seq_code[0:4, i] = np.tile(0.25, 4)
    return np.transpose(seq_code)


#####################
##Load the data######
#####################
def load_data(path):
    df = pd.read_csv(path)

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

    print(x_val.shape)
    print(x_train.shape)
    print(x_test.shape)
    print(y_val.shape)
    print(y_train.shape)
    print(y_test.shape)

    return x_train, x_test, x_val, y_test, y_train, y_val


data_path = '/home/yuxuan/dp/longer_seq_data/eif3a_Full_250.csv'

x_train, x_test, x_val, y_test, y_train, y_val = load_data(data_path)


def retrack(model_loc):
    model = tf.keras.models.load_model(model_loc)
    model.summary()
    print('///////////////////////////')

    with DeepExplain(session=tf.compat.v1.keras.backend.get_session()) as de:
        input_tensor = model.input
        fModel = Model(inputs=input_tensor, outputs=model.output)
        target_tensor = fModel(input_tensor)

        attributions_pos = de.explain('grad*input', target_tensor, input_tensor, xs=x_val[:100], ys=y_val[:100])

    return attributions_pos


def plotpdf(location, weight):
    print('---------------------')
    print(sum(weight)[:, 1])
    print(sum(weight)[:, 2])
    print(sum(weight)[:, 3])
    print(sum(weight)[:, 0])
    a = sum(weight)[:, 0]
    b = sum(weight)[:, 1]
    c = sum(weight)[:, 2]
    d = sum(weight)[:, 3]
    # print(type(a))
    print(len(a))
    all = np.concatenate((a, b, c, d)).tolist()
    # print(type(all))
    # print(all)
    import heapq
    re1 = map(all.index, heapq.nlargest(20, all))
    re2 = heapq.nlargest(20, all)
    re3 = map(all.index, heapq.nsmallest(20, all))
    print(list(re1))
    print(list(re3))
    # print(re2)

    weight[:, 249:251, :] = 0
    print('---------')

    with PdfPages(location) as pdf:
        label = ['A', 'C', 'G', 'T']
        for i in range(4):
            plt.title(label[i])
            plt.bar(np.arange(501) - 250, sum(weight)[:, i])
            pdf.savefig()
            plt.close()
            plt.title(label[i])
            plt.bar(np.arange(101) - 50, sum(weight)[225:326, i])
            pdf.savefig()
            plt.close()

    print('DONE!!!!!!!')


if __name__ == '__main__':
    attributions_pos = retrack('/home/yuxuan/dp/model/eif3a_Full_250_CRNNmodel.h5')

    print('---------BEFORE THE PLOTDF')

    plotpdf('/home/yuxuan/dp/image/eif3a_full_501.pdf', attributions_pos)
