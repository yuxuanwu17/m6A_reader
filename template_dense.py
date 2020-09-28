# %%
# To pkeras_model=None training, we import the necessary functions and submodules from keras
import pandas as pd
import numpy as np
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
import matplotlib
from sklearn.model_selection import train_test_split
# from tensorboard._vendor.tensorflow_serving.apis.classification_pb2 import ClassificationResult

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, precision_recall_curve, auc
from pandas import DataFrame
import h5py

from keras.models import load_model


# %%
def load_data(path):
    df = pd.read_csv(path)
    # print(df)
    n = len(df.columns)
    train = int(n / 2)
    x_train = df.iloc[:, 2:train]

    x_val = df.iloc[:, (train + 1):(n - 1)]
    x_val = DataFrame(x_val)
    x_val = x_val.dropna()
    # print(x_val)

    # x_train = np.expand_dims(x_train, axis=1)
    # x_val = np.expand_dims(x_val, axis=1)

    y_train = df.iloc[:, train:train + 1]
    y_val = df.iloc[:, (n - 1):]
    y_val = DataFrame(y_val)
    y_val = y_val.dropna()
    y_val = DataFrame(y_val, dtype=int)

    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5)

    print(x_val.shape)
    print(x_train.shape)
    print(x_test.shape)
    print(y_val.shape)
    print(y_train.shape)
    print(y_test.shape)

    return x_train, x_test, x_val, y_test, y_train, y_val


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


# %%
def build_model(x_train):
    model = Sequential()
    model.add(Dense(90,
                    kernel_regularizer=regularizers.l2(0.01),
                    input_shape=x_train.shape[1::]))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation("relu"))
    model.add(Dropout(0.25))

    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    # adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
    model.compile(loss='binary_crossentropy', optimizer='Adam',
                  metrics=['accuracy'])
    return model


# %%
def compileModel(model, x_train, x_val, y_val, y_train, gene, condition, encoding):
    model = model
    x_train = x_train
    x_val = x_val
    y_val = y_val
    y_train = y_train
    earlystop = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=10,
                              verbose=1)

    # filepath = '/home/yuxuan/dp/weights/'+gene + '_' + condition + '_' + encoding + "bese_weights.hdf5"
    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, earlystop]

    epoch = 10000
    batchsize = 32

    history = model.fit(x_train,
                        y_train,
                        batch_size=batchsize,
                        epochs=epoch,
                        validation_data=(x_val, y_val),
                        callbacks=callbacks_list)
    modelpath = '/home/yuxuan/dp/model/' + gene + '_' + condition + '_' + encoding + "best_model.h5"
    print(modelpath)
    # modelpath = '/home/yuxuan/dp/testmodel.h5'
    model.save(modelpath)
    # print(filepath)
    return history


# ################################
# print('draw the loss plot')
# ###############################
def lossplot(history, gene, condition, encoding):
    ori_val_Loss = history.history['val_loss']
    loss = history.history['loss']
    epochs = np.arange(len(history.epoch)) + 1
    plt.plot(epochs, ori_val_Loss, label='val loss')
    plt.plot(epochs, loss, label='loss')
    plt.title("Effect of model capacity on validation loss\n")
    plt.xlabel('Epoch #')
    plt.ylabel('Validation Loss')
    plt.legend()
    # plt.show()
    fig_path = '/home/yuxuan/dp/loss_plot/' + gene + '_' + condition + '_' + encoding + '.png'
    plt.savefig(fig_path)
    # plt.savefig('/home/yuxuan/dp/m6aReader/loss_m6areader.png')
    print("")
    print("The loss plot is saved \n")


def roc(model, x_val, y_val, gene, condition, encoding):
    print('Start drawing the roc curve \n')
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    y_pred_keras = model.predict(x_val).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_val, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)

    plt.cla()
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='AUROC (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    # plt.show()
    print('AUROC (area = {:.3f})'.format(auc_keras))
    fig_path = '/home/yuxuan/dp/roc_plot/' + gene + '_' + condition + '_' + encoding + '.png'
    plt.savefig(fig_path)
    auc_keras = '%.3f' % auc_keras
    return float(auc_keras)


def prcurve(model, x_val, y_val, gene, condition, encoding):
    lr_probs = model.predict_proba(x_val)
    lr_precision, lr_recall, _ = precision_recall_curve(y_val, lr_probs)
    lr_auc = auc(lr_recall, lr_precision)

    # summarize scores
    print('PRAUC:  auc=%.3f' % (lr_auc))
    # plot the precision-recall curves
    no_skill = len(y_val[y_val == 1]) / len(y_val)
    pyplot.cla()
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    # pyplot.show()
    fig_path = '/home/yuxuan/dp/pr_plot/' + gene + '_' + condition + '_' + encoding + '.png'
    plt.savefig(fig_path)
    lr_auc = '%.3f' % lr_auc

    return float(lr_auc)


def MCC(model, x_val, y_val):
    from sklearn.metrics import matthews_corrcoef
    yhat = model.predict_classes(x_val)
    mcc = matthews_corrcoef(y_val, yhat)
    print('MCC = {:.3f}'.format(mcc))
    mcc = '%.3f' % mcc
    return float(mcc)


def ACC(model, x_val, y_val):
    from sklearn.metrics import accuracy_score
    yhat = model.predict_classes(x_val)
    acc = accuracy_score(y_val, yhat)
    print('ACC = {:.3f}'.format(acc))
    acc = '%.3f' % acc
    return float(acc)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gene', dest='gene', default=None, type=str, help='select the gene')
    parser.add_argument('-condition', dest='condition', default=None, type=str, help='select full or exon')
    parser.add_argument('-encoding', dest='encoding', default=None, type=str, help='select the encoding method')
    parser.add_argument('-length', dest='length', default=None, type=str, help='specify the sequence length 41/81/121')
    args = parser.parse_args()

    ## assign the input value to variables
    gene = args.gene
    condition = args.condition
    encoding = args.encoding
    length = args.length
    # data_path = '/home/yuxuan/dp/' + gene + '_' + condition + '_' + encoding + '.csv'
    data_path = '/home/yuxuan/dp/{}_{}_{}.csv'.format(gene, condition, encoding)
    # print(data_path)

    x_train, x_test, x_val, y_test, y_train, y_val = load_data(data_path)

    model = build_model(x_train)
    history = compileModel(model, x_train, x_val, y_val, y_train, gene, condition, encoding)

    lossplot(history, gene, condition, encoding)
    auc = roc(model, x_val, y_val, gene, condition, encoding)
    prauc = prcurve(model, x_val, y_val, gene, condition, encoding)
    mcc = MCC(model, x_val, y_val)
    acc = ACC(model, x_val, y_val)
    results = np.array([auc, prauc, mcc, acc])
    # print(results)

    mtx_path = '/home/yuxuan/dp/storeMatrix/{}_{}_{}Den.csv'.format(gene, condition, encoding)
    np.savetxt(mtx_path, results, delimiter=',', fmt='%.3f')

    ######load model######
    ######################
    # modelpath = '/home/yuxuan/dp/model/' + gene + '_' + condition + '_' + encoding + "best_model.h5"
    # model = load_model(modelpath)
    # model.summary()
    # yhat = model.predict(x_test, verbose=0)
    # print(ClassificationResult(y_test, yhat))
    # print(yhat)


if __name__ == '__main__':
    main()
