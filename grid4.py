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
from keras.wrappers.scikit_learn import KerasClassifier
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, precision_recall_curve, auc

# plt.use('Agg')
from sklearn.model_selection import GridSearchCV, train_test_split

K.set_image_data_format('channels_last')

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


#####################
##Load the data######
#####################
def load_data():
    df = pd.read_csv("/home/yuxuan/dp/eif3a_full_conpositionDen10.csv")
    # print(df)
    n = len(df.columns)
    train = int(n / 2)
    x_train = df.iloc[:, 2:train]

    x_val = df.iloc[:, (train + 1):(n - 1)]
    x_val = pd.DataFrame(x_val)
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


##########################################################
#####Define the model architecture in keras###############
#########################################################

def build_model(input_shape=[10], optimizer='adam', init_mode='uniform', dropout_rate=0.0, weight_constraint=0,
                neurons=300):
    model = Sequential()
    model.add(
        Dense(1200, kernel_initializer=init_mode,
              kernel_regularizer=regularizers.l2(0.01),
              kernel_constraint=maxnorm(weight_constraint),
              input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons))
    model.add(Activation("softmax"))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    # print('model_complete')
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def main():
    x_train, x_test, x_val, y_test, y_train, y_val = load_data()
    # print(x_train.shape[1::])
    model = KerasClassifier(build_fn=build_model, input_shape=x_train.shape[1::], verbose=0)

    # 定义网格搜索参数
    # batch_size = [16, 32, 64, 128, 256]
    batch_size = [128]
    epochs = [100]
    # epochs = [50, 80, 100, 200, 300]
    # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    optimizer = ['Adam']
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal',
                 'glorot_uniform', 'he_normal', 'he_uniform']
    # init_mode = ['normal']
    weight_constraint = [1]
    # weight_constraint = [1,2,3,4,5]
    # dropout_rate = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]  ## 建议最好是从0。1-0。9都包括
    # neurons = [1, 5, 10, 15, 20, 25, 30]
    dropout_rate = [0.1]
    neurons = [1]

    param_grid = dict(batch_size=batch_size,
                      epochs=epochs,
                      optimizer=optimizer,
                      init_mode=init_mode,
                      weight_constraint=weight_constraint,
                      dropout_rate=dropout_rate,
                      neurons=neurons)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)  ##那个n-jobs 和 pre-dispatch暂时搞不定
    grid_result = grid.fit(x_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


if __name__ == '__main__':
    main()
