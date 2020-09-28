# To pkeras_model=None training, we import the necessary functions and submodules from keras
import matplotlib
import pandas as pd
import numpy as np
from keras.layers import Bidirectional, LSTM
from pandas import DataFrame
from keras.models import Sequential
from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import Adadelta, SGD, RMSprop;
import keras.losses;
from keras.constraints import maxnorm;
from keras.utils import normalize, to_categorical
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from keras import backend as K

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, precision_recall_curve, auc

from sklearn.model_selection import train_test_split

K.set_image_data_format('channels_last')

import os

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


def roc(model, x_val, y_val, gene, condition, length):
    print('Start drawing the roc curve \n')
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    y_pred_keras = model.predict(x_val).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_val, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)

##改动的位置

    plt.cla()
    plt.figure(figsize=(4,3),dpi=1000)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='AUROC (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    # plt.show()
    print('AUROC (area = {:.3f})'.format(auc_keras))
    plt.savefig('/home/yuxuan/dp/onehot/{}_{}_{}_ROC(RNN)_test5.pdf'.format(gene, condition, length), format='pdf')
    return auc_keras


def prcurve(model, x_val, y_val, gene, condition, length):
    lr_probs = model.predict_proba(x_val)
    lr_precision, lr_recall, _ = precision_recall_curve(y_val, lr_probs)
    lr_auc = auc(lr_recall, lr_precision)

    # summarize scores
    print('PRAUC:  auc=%.3f' % (lr_auc))
    # plot the precision-recall curves
    no_skill = len(y_val[y_val == 1]) / len(y_val)
    pyplot.cla()
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.plot(lr_recall, lr_precision, marker='.', label='CNN+RNN')
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    # pyplot.show()
    plt.savefig('/home/yuxuan/dp/onehot/{}_{}_{}_PRAUC(RNN)_test.pdf'.format(gene, condition, length))
    return lr_auc


def main():
    # gene = ['eif3a', 'YTHDF3','YTHDF1','YTHDF2']
    # gene = ['YTHDF1']
    gene = ['eif3a']
    # gene = ['YTHDC1','YTHDC2']
    # gene = ['YTHDC1']

    # condition = ['Exon', 'Full']
    condition = ['Full']
    # length = ['1000', '500', '250', '125']
    length = ['125']

    for x in gene:
        # print(gene)
        for y in condition:
            # print(gene)
            # print(condition)
            for z in length:
                # data_path = '/home/yuxuan/dp/longer_seq_data/YTHDF1/{}_{}_{}.csv'.format(x, y, z)
                data_path = '/home/yuxuan/dp/longer_seq_data/{}_{}_{}.csv'.format(x, y, z)
                print(data_path)

                x_train, x_test, x_val, y_test, y_train, y_val = load_data(data_path)
                model = build_model(x_train)
                history = compileModel(model, x_train, x_val, y_val, y_train, x, y, z)
                lossplot(history, x, y, z)
                auc = roc(model, x_val, y_val, x, y, z)
                prauc = prcurve(model, x_val, y_val, x, y, z)
                mcc = MCC(model, x_val, y_val)
                acc = ACC(model, x_val, y_val)
                results = np.array([auc, prauc, mcc, acc])
                np.savetxt('/home/yuxuan/dp/CNN/longseq/{}_{}_{}(RNN)_test.csv'.format(x, y, z), results, delimiter=',',
                           fmt='%.3f')


if __name__ == '__main__':
    main()
