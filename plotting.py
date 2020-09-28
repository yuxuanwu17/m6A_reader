import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve, auc
import numpy as np


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


def roc(model, x_val, y_val, gene, condition, length):
    print('Start drawing the roc curve \n')
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    y_pred_keras = model.predict(x_val).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_val, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)

    plt.cla()
    plt.figure(figsize=(4, 3), dpi=1000)
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
