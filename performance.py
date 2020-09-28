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