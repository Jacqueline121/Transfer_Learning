from __future__ import division
import numpy as np
from keras.models import load_model, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense
import utils
from keras.optimizers import SGD


def get_basemodel():
    model_path = 'model.hdf5'
    mvgg_model = load_model(model_path)
    base_model = Model(inputs=mvgg_model.input, outputs=mvgg_model.get_layer('flatten_1').output)
    return base_model


def get_model(base_model):
    x = base_model.output
    x = Dense(1024, activation='relu')(x)
    prediction = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=prediction)
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model


if __name__ == '__main__':
    x_test, y_test = utils.get_data('/home/ilab/data10/test', 2)
    x_train, y_train = utils.get_data('/home/ilab/data10/augdata/train', 2)
    y_train = utils.one_hot_encode(y_train)
    y_test = utils.one_hot_encode(y_test)
    base_model = get_basemodel()
    model = get_model(base_model)
    for layer in base_model.layers:
        layer.trainable = False
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








