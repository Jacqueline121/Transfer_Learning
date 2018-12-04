from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.optimizers import SGD


def cnn_model(image_wdith, image_height, channels, classes):

    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                     input_shape=(image_wdith, image_height, channels)))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(classes, activation='sigmoid'))

    sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    return model