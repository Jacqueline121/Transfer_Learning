from __future__ import division
import numpy as np
from PIL import Image
import os
from linear_classifier import LinearSVM
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os


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

    num_training = 19000
    num_validation = 385
    num_test = 168

    # validation set will be num_validation points from the original training set
    mask = range(num_training, num_training + num_validation)
    x_val = x_train[mask]
    y_val = y_train[mask]

    # training set will be the first num_training points from the original training set
    mask = range(num_training)
    x_train = x_train[mask]
    y_train = y_train[mask]

    # use the first num_test points of the original test set as our test set
    mask = range(num_test)
    x_test = x_test[mask]
    y_test = y_test[mask]

    # reshape the image data into rows
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_val = np.reshape(x_val, (x_val.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))

    # subtract the mean image
    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_val -= mean_image
    x_test -= mean_image

    x_train = np.hstack([x_train, np.ones((x_train.shape[0], 1))]).T
    x_val = np.hstack([x_val, np.ones((x_val.shape[0], 1))]).T
    x_test = np.hstack([x_test, np.ones((x_test.shape[0], 1))]).T

    svm = LinearSVM()

    # find the best learning_rate and regularization_strengths

    learning_rates = [5e-8, 1e-7, 5e-7, 1e-6]
    regularization_strengths = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5]

    results = {}
    # The highest validation accuracy that we have seen so far
    best_val = -1

    # The LinearSVM object that achieved the highest validation rate
    best_svm = None

    for strength in regularization_strengths:
        for rate in learning_rates:
            svm = LinearSVM()
            svm.train(x_train, y_train, learning_rate=rate, reg=strength, num_iters=1500, verbose=True)
            y_train_pred = svm.predict(x_train)
            train_accuracy = np.mean(y_train == y_train_pred)
            y_valid_pred = svm.predict(x_val)
            val_accuracy = np.mean(y_val == y_valid_pred)
            results[(rate, strength)] = (train_accuracy, val_accuracy)

            if val_accuracy > best_val:
                best_val = val_accuracy
                best_svm = svm

    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
            lr, reg, train_accuracy, val_accuracy)

    print 'best validation accuracy achieved during cross-validation: %f' % best_val

    y_test_pred = best_svm.predict(x_test)
    test_accuracy = np.mean(y_test == y_test_pred)
    print 'linear SVM on raw pixels final test set accuracy: %f' % test_accuracy

    y_test_pred = one_hot_encode(y_test_pred)
    y_test = one_hot_encode(y_test)
    acc = get_accuracy(y_test, y_test_pred)
    sen, spe = get_sen_spe(y_test, y_test_pred)
    fpr, tpr, AUC = get_roc_auc(y_test_pred[:, 0], y_test[:, 0])
    np.savetxt('svm.txt', (fpr, tpr))
    a = np.loadtxt('svm.txt')
    plot_roc_curve(a[0, :], a[1, :], AUC)
    print 'The accuracy of this model is %f' % acc
    print 'The sensitive of this model is %f ' % sen
    print 'The specificity of this model is %f' % spe
    print 'The AUC of this model is %f' % AUC