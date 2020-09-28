# To pkeras_model=None training, we import the necessary functions and submodules from keras
import matplotlib
import pandas as pd
import numpy as np
from keras.layers import Bidirectional, LSTM
from pandas import DataFrame
from keras.models import Sequential
from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras import regularizers
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from keras import backend as K
from one_hot import seq_to_mat
from plotting import lossplot, roc, prcurve

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, precision_recall_curve, auc

from sklearn.model_selection import train_test_split

K.set_image_data_format('channels_last')

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


#####################
##Load the data######
#####################
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

    print(x_val.shape)
    print(x_train.shape)
    print(x_test.shape)
    print(y_val.shape)
    print(y_train.shape)
    print(y_test.shape)

    return x_train, x_test, x_val, y_test, y_train, y_val


##########################################################
#####Define the model architecture in keras###############
#########################################################

def build_model(x_train):
    one_filter_keras_model = Sequential()
    one_filter_keras_model.add(
        Conv1D(filters=90, kernel_size=5, padding="valid", kernel_regularizer=regularizers.l2(0.01),
               input_shape=x_train.shape[1::]))
    one_filter_keras_model.add(Activation('relu'))
    one_filter_keras_model.add(MaxPooling1D(pool_size=4, strides=2))
    one_filter_keras_model.add(Dropout(0.25))

    one_filter_keras_model.add(Bidirectional(LSTM(35, return_sequences=True)))
    one_filter_keras_model.add(Dropout(0.25))
    one_filter_keras_model.add(Flatten())

    one_filter_keras_model.add(Dense(1))
    one_filter_keras_model.add(Activation("sigmoid"))
    one_filter_keras_model.summary()

    one_filter_keras_model.compile(loss='binary_crossentropy', optimizer='adam',
                                   metrics=['accuracy'])
    return one_filter_keras_model


def compileModel(model, x_train, x_val, y_val, y_train, gene, condition, length):
    model = model
    x_train = x_train
    x_val = x_val
    y_val = y_val
    y_train = y_train
    earlystop = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=10,
                              verbose=1)
    ##########################################

    # file path need to be further explored
    ##########################################
    filepath = "weights.best_encoding.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, earlystop]

    epoch = 1000
    batchsize = 128

    history = model.fit(x_train, y_train, batch_size=batchsize, epochs=epoch,
                        validation_data=(x_val, y_val),
                        callbacks=callbacks_list)
    modelpath = '/home/yuxuan/dp/model/{}_{}_{}_CRNNmodel_test.h5'.format(gene, condition, length)
    model.save(modelpath)

    return history


# ################################
# print('draw the loss plot')
# ###############################
def lossplot(history, gene, condition, length):
    ori_val_Loss = history.history['val_loss']
    loss = history.history['loss']
    epochs = np.arange(len(history.epoch)) + 1
    plt.cla()
    plt.plot(epochs, ori_val_Loss, label='val loss')
    plt.plot(epochs, loss, label='loss')
    plt.title("Effect of model capacity on validation loss\n")
    plt.xlabel('Epoch #')
    plt.ylabel('Validation Loss')
    plt.legend()
    # plt.show()
    plt.savefig('/home/yuxuan/dp/onehot/{}_{}_{}_lossplot(RNN)_test.png'.format(gene, condition, length))
    print("")
    print("The loss plot is saved \n")


def MCC(model, x_val, y_val):
    from sklearn.metrics import matthews_corrcoef
    yhat = model.predict_classes(x_val)
    mcc = matthews_corrcoef(y_val, yhat)
    print('MCC = {:.3f})'.format(mcc))
    return mcc


def ACC(model, x_val, y_val):
    from sklearn.metrics import accuracy_score
    yhat = model.predict_classes(x_val)
    acc = accuracy_score(y_val, yhat)
    print('ACC = {:.3f})'.format(acc))
    return acc




def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gene', dest='gene', default=None, type=str, help='select the gene')
    parser.add_argument('-condition', dest='condition', default=None, type=str, help='select full or exon')
    parser.add_argument('-length', dest='length', default=None, type=str, help='specify the two ends sequence length 125/250/500/1000')
    args = parser.parse_args()

    ## assign the input value to variables
    gene = args.gene
    condition = args.condition
    length = args.length

    data_path = '/home/yuxuan/dp/longer_seq_data/{}_{}_{}.csv'.format(gene, condition, length)

    x_train, x_test, x_val, y_test, y_train, y_val = load_data(data_path)
    model = build_model(x_train)
    history = compileModel(model, x_train, x_val, y_val, y_train, gene, condition, length)
    lossplot(history, gene,condition, length)
    auc = roc(model, x_val, y_val, gene, condition, length)
    prauc = prcurve(model, x_val, y_val, gene, condition, length)
    mcc = MCC(model, x_val, y_val)
    acc = ACC(model, x_val, y_val)
    results = np.array([auc, prauc, mcc, acc])
    np.savetxt('/home/yuxuan/dp/CNN/longseq/{}_{}_{}(RNN)_test.csv'.format(gene, condition, length), results, delimiter=',',
               fmt='%.3f')


if __name__ == '__main__':
    main()
