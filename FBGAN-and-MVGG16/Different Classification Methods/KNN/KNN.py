from __future__ import division
import numpy as np
from PIL import Image
import os
import k_nearest_neighbor
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def get_data(file_path, num_type):
    print ('lode data from files...')
    x = []
    y = []
    for i in range(num_type):
        img_path = file_path + '/' + str(i)
        imgs = os.listdir(img_path)
        for img in imgs:
            image = Image.open(img_path + '/' + img)
            x.append(np.asarray(image, dtype='float32'))
            y.append(i)

    x_data = np.array(x)
    y_data = np.array(y)

    np.random.seed(1)
    permutation = np.random.permutation(x_data.shape[0])
    shuffled_x = x_data[permutation, :, :, :]
    shuffled_y = y_data[permutation]

    return shuffled_x, shuffled_y


def get_accuracy(y_test, y_result):
    max_idx_test = np.argmax(y_test, axis=1)
    max_idx_result = np.argmax(y_result, axis=1)
    total = y_test.shape[0]
    count = 0

    for i in range(total):
        if max_idx_result[i] == max_idx_test[i]:
            count += 1
    return float(count)/total


def get_sen_spe(y_test, y_result):
    max_idx_test = np.argmax(y_test, axis=1)
    max_idx_result = np.argmax(y_result, axis=1)
    num = y_test.shape[0]
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(num):
        if max_idx_test[i] == 0:
            if max_idx_result[i] == 0:
                TN = TN + 1
            else:
                FP = FP + 1
        else:
            if max_idx_result[i] == 0:
                FN = FN + 1
            else:
                TP = TP + 1
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    return sensitivity, specificity


def get_roc_auc(score, label):
    fpr, tpr, threshold = roc_curve(label, score, drop_intermediate =False)
    AUC = auc(fpr, tpr)
    return fpr, tpr, AUC


def plot_roc_curve(fpr, tpr, auc):
    plt.figure()
    lw = 1
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def one_hot_encode(y):
    l = y.shape[0]
    one_hot_y = np.empty((l, 2))
    for i in range(l):
        if y[i] == 0:
            one_hot_y[i, 0] = 0
            one_hot_y[i, 1] = 1
        elif y[i] == 1:
            one_hot_y[i, 0] = 1
            one_hot_y[i, 1] = 0
    return one_hot_y


if __name__ == '__main__':
    x_test, y_test = get_data('../Data/test', 2)
    x_train, y_train = get_data('../Data/train', 2)
    num_test = x_test.shape[0]
    classifier = k_nearest_neighbor.KNearestNeighbor()
    classifier.train(x_train, y_train)
    dists = classifier.compute_distances_two_loops(x_test)
    y_test_pred = classifier.predict_labels(dists, k=5)
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)
    y_test_pred = one_hot_encode(y_test_pred)
    y_test = one_hot_encode(y_test)
    acc = get_accuracy(y_test, y_test_pred)
    sen, spe = get_sen_spe(y_test, y_test_pred)
    fpr, tpr, AUC = get_roc_auc(y_test_pred[:, 0], y_test[:, 0])
    np.savetxt('knn.txt', (fpr, tpr))
    a = np.loadtxt('knn.txt')
    plot_roc_curve(a[0, :], a[1, :], AUC)
    print 'The accuracy of this model is %f' % acc
    print 'The sensitive of this model is %f ' % sen
    print 'The specificity of this model is %f' % spe
    print 'The AUC of this model is %f' % AUC