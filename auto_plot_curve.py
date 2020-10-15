import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import average_precision_score, auc
from load_data import load_data
from model import build_model, compileModel, build_model_CNN
from numpy import interp
from itertools import cycle
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    # gene = ['eif3a', 'YTHDF3','YTHDF1','YTHDF2']
    # gene = ['YTHDF1']
    gene = ['eif3a']
    # gene = ['YTHDC1','YTHDC2']
    # gene = ['YTHDC1']

    # condition = ['Exon', 'Full']
    condition = ['Full']
    # length = ['1000', '500', '250', '125']
    length = ['125', '250', '500', '1000']
    mode = 'CNN+RNN'

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    average_precision = dict()
    name = dict()
    epoch = 0
    lw = 2

    for x in gene:
        for y in condition:
            for z in length:

                data_path = '/home/yuxuan/dp/longer_seq_data/{}_{}_{}.csv'.format(x, y, z)

                _, x_test, _, y_test, _, _ = load_data(data_path)

                if mode == 'CNN+RNN':
                    model = build_model(x_test)
                    checkpoint_path = '/home/yuxuan/dp/model/{}_{}_{}_CRNNmodel_test.h5'.format(x, y, z)
                else:
                    model = build_model_CNN(x_test)
                    checkpoint_path = '/home/yuxuan/dp/model/{}_{}_{}_best_model.h5'.format(x, y, z)
                    print(checkpoint_path)

                model.load_weights(checkpoint_path)
                y_score = model.predict(x_test)

                ## PR curve
                precision[epoch], recall[epoch], _ = precision_recall_curve(y_true=y_test, probas_pred=y_score)
                average_precision[epoch] = average_precision_score(y_true=y_test, y_score=y_score)


                ## ROC curve
                fpr[epoch], tpr[epoch], _ = roc_curve(y_true=y_test, y_score=y_score)
                roc_auc[epoch] = auc(fpr[epoch], tpr[epoch])
                name[epoch]='{}_{}_{}'.format(x.upper(),y,z)
                epoch = epoch + 1

    ## ROC plotting
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(epoch), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of {}(area = {:.2f})'
                       ''.format(name[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(' ROC plots of EIF3A Full transcripts in different length')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig('/home/yuxuan/dp_m6a_org/plot/ROC(RNN_all).png',
                format='png')
    plt.cla()
    plt.figure(figsize=(7, 8))

    ## PR curve plotting
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    lines = []
    labels = []

    for i, color in zip(range(epoch), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for {0} (area = {1:0.2f})'
                      ''.format(name[i], average_precision[i]))


    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve to EIF3A Full transcript in different lengths')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.savefig('/home/yuxuan/dp_m6a_org/plot/PR_Curve(RNN_all).png',
                format='png')


if __name__ == '__main__':
    main()
