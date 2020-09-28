# To pkeras_model=None training, we import the necessary functions and submodules from keras


import numpy as np
from keras.layers import Bidirectional, LSTM

from keras.models import Sequential
from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras import regularizers
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from keras import backend as K

from plotting import lossplot, roc, prcurve
from sklearn.model_selection import train_test_split
from performance import ACC, MCC

K.set_image_data_format('channels_last')

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "5"




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



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gene', dest='gene', default=None, type=str, help='select the gene')
    parser.add_argument('-condition', dest='condition', default=None, type=str, help='select full or exon')
    parser.add_argument('-length', dest='length', default=None, type=str,
                        help='specify the two ends sequence length 125/250/500/1000')
    args = parser.parse_args()

    ## assign the input value to variables
    gene = args.gene
    condition = args.condition
    length = args.length

    data_path = '/home/yuxuan/dp/longer_seq_data/{}_{}_{}.csv'.format(gene, condition, length)

    x_train, x_test, x_val, y_test, y_train, y_val = load_data(data_path)
    model = build_model(x_train)
    history = compileModel(model, x_train, x_val, y_val, y_train, gene, condition, length)
    lossplot(history, gene, condition, length)
    auc = roc(model, x_val, y_val, gene, condition, length)
    prauc = prcurve(model, x_val, y_val, gene, condition, length)
    mcc = MCC(model, x_val, y_val)
    acc = ACC(model, x_val, y_val)
    results = np.array([auc, prauc, mcc, acc])
    np.savetxt('/home/yuxuan/dp/CNN/longseq/{}_{}_{}(RNN)_test.csv'.format(gene, condition, length), results,
               delimiter=',',
               fmt='%.3f')


if __name__ == '__main__':
    main()
