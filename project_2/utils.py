import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def cluster_score(labels,pred,p = False):
    h = metrics.homogeneity_score(labels, pred)
    c = metrics.completeness_score(labels, pred)
    v = metrics.v_measure_score(labels, pred)
    ARI = metrics.adjusted_rand_score(labels,pred)
    m = metrics.mutual_info_score(labels,pred)

    if p == True:
        print("Homogeneity: %0.3f" % h)
        print("Completeness: %0.3f" % c)
        print("V-measure: %0.3f" % v)
        print("Adjusted Rand-Index: %.3f"% ARI)
        print("mutual_info_score: %.3f"% m)
    return [h,c,v,ARI,m]

def compare_result(Best_LSI,lables,pred):
    cnf_matrix = metrics.confusion_matrix(lables, pred)

    plot_confusion_matrix(cnf_matrix, classes=['tech','rec'],
                                title='Confusion matrix, without normalization')        
    cluster_score(lables,pred,p = True)

    x1 = Best_LSI[pred == 0][:, 0]
    y1 = Best_LSI[pred == 0][:, 1]
    x2 = Best_LSI[pred == 1][:, 0]
    y2 = Best_LSI[pred == 1][:, 1]

    x3 = Best_LSI[lables == 0][:, 0]
    y3 = Best_LSI[lables == 0][:, 1]
    x4 = Best_LSI[lables == 1][:, 0]
    y4 = Best_LSI[lables == 1][:, 1]

    plt.figure()
    plt.plot(x1,y1,'b.')
    plt.plot(x2,y2,'g.')
    plt.title("clustering classification")

    plt.figure()
    plt.plot(x3,y3,'b.')
    plt.plot(x4,y4,'g.')
    plt.title("actual classification")


















    
