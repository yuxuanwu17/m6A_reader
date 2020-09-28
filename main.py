import numpy as np
from keras import backend as K
from plotting import lossplot, roc, prcurve
from performance import ACC, MCC
from load_data import load_data
from model import build_model,compileModel,build_model_CNN
K.set_image_data_format('channels_last')

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gene', dest='gene', default=None, type=str, help='select the gene')
    parser.add_argument('-condition', dest='condition', default=None, type=str, help='select full or exon')
    parser.add_argument('-length', dest='length', default=None, type=str,
                        help='specify the two ends sequence length 125/250/500/1000')
    parser.add_argument('-mode', default=None, type=str, help='select your framework, CNN or CNN+RNN')
    args = parser.parse_args()

    ## assign the input value to variables
    gene = args.gene
    condition = args.condition
    length = args.length
    mode = args.mode

    data_path = '/home/yuxuan/dp/longer_seq_data/{}_{}_{}.csv'.format(gene, condition, length)

    x_train, x_test, x_val, y_test, y_train, y_val = load_data(data_path)
    if mode =='CNN+RNN':
        model = build_model(x_train)
    else:
        model =build_model_CNN(x_train)
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
