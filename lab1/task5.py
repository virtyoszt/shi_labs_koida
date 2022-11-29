import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

df = pd.read_csv('data_metrics.csv')
df.head()
thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
df.head()
print(confusion_matrix(df.actual_label.values, df.predicted_RF.values))


def find_tp(y_true, y_pred):
    # counts the number of true positives (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))


def find_fn(y_true, y_pred):
    # counts the number of false negatives (y_true = 1, y_pred = 0)
    return sum((y_true == 1) & (y_pred == 0))


def find_fp(y_true, y_pred):
    # counts the number of false positives (y_true = 0, y_pred = 1)
    return sum((y_true == 0) & (y_pred == 1))


def find_tn(y_true, y_pred):
    # counts the number of true negatives (y_true = 0, y_pred = 0)
    return sum((y_true == 0) & (y_pred == 0))


print('TP:', find_tp(df.actual_label.values,
                     df.predicted_RF.values))
print('FN:', find_fn(df.actual_label.values,
                     df.predicted_RF.values))
print('FP:', find_fp(df.actual_label.values,
                     df.predicted_RF.values))
print('TN:', find_tn(df.actual_label.values,
                     df.predicted_RF.values))


def find_conf_matrix_values(y_true, y_pred):
    # calculate TP, FN, FP, TN
    tp = find_tp(y_true, y_pred)
    fn = find_fn(y_true, y_pred)
    fp = find_fp(y_true, y_pred)
    tn = find_tn(y_true, y_pred)
    return tp, fn, fp, tn


def my_confusion_matrix(y_true, y_pred):
    tp, fn, fp, tn = find_conf_matrix_values(y_true, y_pred)
    return np.array([[tn, fp], [fn, tp]])


my_confusion_matrix(df.actual_label.values, df.predicted_RF.values)
assert np.array_equal(my_confusion_matrix(df.actual_label.values, df.predicted_RF.values), confusion_matrix(
    df.actual_label.values, df.predicted_RF.values)), 'my_confusion_matrix() is not correct for rf'
assert np.array_equal(my_confusion_matrix(df.actual_label.values, df.predicted_LR.values),
                      confusion_matrix(df.actual_label.values, df.predicted_LR.values)), \
    'my_confusion_matrix() is not correct for lr'
print(accuracy_score(df.actual_label.values, df.predicted_RF.values))


# calculates the fraction of samples

def my_accuracy_score(y_true, y_pred):
    tp, fn, fp, tn = find_conf_matrix_values(y_true, y_pred)
    return (tp + tn) / (tp + tn + fp + fn)


assert my_accuracy_score(df.actual_label.values, df.predicted_LR.values) == accuracy_score(df.actual_label.values,
                                                                                           df.predicted_LR.values), \
    'my_accuracy_score failed on lr'
print('Accuracy RF: % .3f ' % (my_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print(recall_score(df.actual_label.values, df.predicted_RF.values))
f1_score(df.actual_label.values, df.predicted_RF.values)

print(recall_score(df.actual_label.values, df.predicted_RF.values))


def my_recall_score(y_true, y_pred):
    # calculates the fraction of positive samples predicted correctly
    tp, fn, fp, tn = find_conf_matrix_values(y_true, y_pred)
    return tp / (tp + fn)


assert my_recall_score(df.actual_label.values,
                       df.predicted_LR.values) == recall_score(df.actual_label.values, df.predicted_LR.values), \
    'MyAccuracyScore failed on LR'

print('Recall RF: %.3f' % (my_recall_score(df.actual_label.values,
                                           df.predicted_RF.values)))
print('Recall LR: %.3f' % (my_recall_score(df.actual_label.values,
                                           df.predicted_LR.values)))
precision_score(df.actual_label.values, df.predicted_RF.values)


def my_precision_score(y_true, y_pred):
    # calculates the fraction of predicted positives samples that are actually positive
    tp, fn, fp, tn = find_conf_matrix_values(y_true, y_pred)
    return tp / (tp + fp)


assert my_precision_score(df.actual_label.values, df.predicted_LR.values) == precision_score(
    df.actual_label.values,
    df.predicted_LR.values), 'my_accuracy_score failed on LR'
print('Precision RF: %.3f' % (my_precision_score(df.actual_label.values,
                                                 df.predicted_RF.values)))
print('Precision LR: %.3f' % (my_precision_score(df.actual_label.values,
                                                 df.predicted_LR.values)))
f1_score(df.actual_label.values, df.predicted_RF.values)


def my_f1_score(y_true, y_pred):
    # calculates the F1 score
    recall = my_recall_score(y_true, y_pred)
    precision = my_precision_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)


assert my_f1_score(df.actual_label.values, df.predicted_LR.values) == f1_score(df.actual_label.values,
                                                                               df.predicted_LR.values), \
    'my_accuracy_score failed on LR'
print('F1 RF: %.3f' % (my_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 LR: %.3f' % (my_f1_score(df.actual_label.values, df.predicted_LR.values)))
print('scores with threshold = 0.5')
print('Accuracy RF: % .3f' % (my_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall RF: %.3f' % (my_recall_score(df.actual_label.values,
                                           df.predicted_RF.values)))
print('Precision RF: % .3f' % (my_precision_score(df.actual_label.values,
                                                  df.predicted_RF.values)))
print('F1 RF: %.3f' % (my_f1_score(df.actual_label.values, df.predicted_RF.values)))

print('')
print('scores with threshold = 0.25')
print('Accuracy RF: % .3f' % (
    my_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Recall RF: %.3f' % (my_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Precision RF: %.3f' % (
    my_precision_score(df.actual_label.values, (df.model_RF >=
                                                0.25).astype('int').values)))
print('F1 RF: %.3f' % (my_f1_score(df.actual_label.values, (df.model_RF >=
                                                            0.25).astype('int').values)))
fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values,
                                          df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(
    df.actual_label.values, df.model_LR.values)
plt.plot(fpr_RF, tpr_RF, 'r-', label='RF')
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR')
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print('AUC RF:%.3f' % auc_RF)
print('AUC LR:%.3f' % auc_LR)
plt.plot(fpr_RF, tpr_RF, 'r-', label='RF AUC: %.3f' % auc_RF)
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR AUC: %.3f' % auc_LR)
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
