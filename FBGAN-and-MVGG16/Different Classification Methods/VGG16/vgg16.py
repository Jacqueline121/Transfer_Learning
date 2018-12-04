from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import SGD


def vgg_model(image_weight, image_height, channels, classes):
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(image_weight, image_height, channels)))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(classes, activation='sigmoid'))

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model









