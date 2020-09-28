import matplotlib
from sklearn.model_selection import train_test_split

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tensorflow as tf
import numpy as np

from tensorflow.python.keras.models import Sequential, Model
import pandas as pd
from deepexplain.tensorflow import DeepExplain

import os
from pandas import DataFrame

from one_hot import seq_to_mat
from load_data import load_data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


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
    data_path = '/home/yuxuan/dp/longer_seq_data/eif3a_Full_250.csv'
    x_train, x_test, x_val, y_test, y_train, y_val = load_data(data_path)
    attributions_pos = retrack('/home/yuxuan/dp/model/eif3a_Full_250_CRNNmodel.h5')
    print('---------BEFORE THE PLOTDF')
    plotpdf('/home/yuxuan/dp_m6a_org/eif3a_full_501.pdf', attributions_pos)
