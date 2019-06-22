from __future__ import division
import numpy as np
from keras.models import load_model, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import utils


def get_model():
    model_path = 'model.hdf5'
    model = load_model(model_path)
    return model



if __name__ == '__main__':
    x_test, y_test = utils.get_data('/home/ilab/data10/test', 2)
    x_train, y_train = utils.get_data('/home/ilab/data10/augdata/train', 2)
    y_train = utils.one_hot_encode(y_train)
    y_test = utils.one_hot_encode(y_test)
    model = get_model()
    early_stop = EarlyStopping(monitor='val_acc', verbose=1, patience=10, mode='max')
    save_best = ModelCheckpoint('mvgg_best_model3200.hdf5', verbose=1, save_best_only=True)
    model.fit(x_train, y_train, epochs=200, batch_size=64, validation_split=0.1, callbacks=[save_best, early_stop])
    model = load_model('mvgg_best_model3200.hdf5')
    score = model.evaluate(x_test, y_test, batch_size=64)
    result = model.predict(x_test)
    acc = utils.get_accuracy(y_test, result)
    sen, spe = utils.get_sen_spe(y_test, result)
    fpr, tpr, AUC = utils.get_roc_auc(result[:, 0], y_test[:, 0])
    np.savetxt('mvgg16.txt', (fpr, tpr))
    a = np.loadtxt('mvgg16.txt')
    utils.plot_roc_curve(a[0, :], a[1, :], AUC)

    print 'The accuracy of this model is %f' % acc
    print 'The sensitive of this model is %f ' % sen
    print 'The specificity of this model is %f' % spe
    print 'The AUC of this model is %f' % AUC








