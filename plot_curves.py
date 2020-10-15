import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import average_precision_score, auc
from load_data import load_data
from model import build_model,compileModel,build_model_CNN
from scipy import interpolate

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

    _, x_test, _, y_test, _, _ = load_data(data_path)

    if mode =='CNN+RNN':
        model = build_model(x_test)
        checkpoint_path = '/home/yuxuan/dp/model/{}_{}_{}_CRNNmodel_test.h5'.format(gene, condition, length)
    else:
        model =build_model_CNN(x_test)
        checkpoint_path = '/home/yuxuan/dp/model/{}_{}_{}_best_model.h5'.format(gene, condition, length)
        print(checkpoint_path)

    model.load_weights(checkpoint_path)
    y_score = model.predict(x_test)
    precision, recall, _ = precision_recall_curve(y_true=y_test, probas_pred=y_score)
    average_precision = average_precision_score(y_true=y_test, y_score=y_score)

## ROC curve
    fpr, tpr, _ = roc_curve(y_true=y_test, y_score=y_score)
    roc_auc = auc(fpr, tpr)

if __name__ == '__main__':
    main()