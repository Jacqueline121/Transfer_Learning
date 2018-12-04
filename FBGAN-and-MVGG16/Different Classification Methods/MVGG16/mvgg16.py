from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Input
from keras.layers.core import Flatten, Dense, Activation
from keras.optimizers import SGD


def multi_block(x, stride):
    out = Conv2D(128, kernel_size=3, strides=stride[0])(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(128, kernel_size=1, strides=stride[1])(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(512, kernel_size=1, strides=stride[2])(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    return out


def mvgg_model(image_width, image_height, channels, classes):

    inp = Input(shape=(image_width, image_height, channels))
    out = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(inp)
    out = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='block1')(out)
    out = MaxPooling2D(pool_size=2, strides=2)(out)
    stride = [4, 2, 2]
    block1 = multi_block(out, stride)

    out = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(out)
    out = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', name='block2')(out)
    out = MaxPooling2D(pool_size=2, strides=2)(out)
    stride = [4, 2, 1]
    block2 = multi_block(out, stride)

    out = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(out)
    out = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(out)
    out = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', name='block3')(out)
    out = MaxPooling2D(pool_size=2, strides=2)(out)
    stride = [2, 2, 1]
    block3 = multi_block(out, stride)

    out = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(out)
    out = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(out)
    out = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu', name='block4')(out)
    out = MaxPooling2D(pool_size=2, strides=2)(out)
    stride = [1, 1, 1]
    block4 = multi_block(out, stride)

    out = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(out)
    out = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(out)
    out = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu', name='block5')(out)
    out = MaxPooling2D(pool_size=2, strides=2)(out)
    out = Concatenate()([out, block1, block2, block3, block4])

    out = Flatten()(out)

    out = Dense(4096, activation='relu')(out)
    out = Dense(4096, activation='relu', name='dense')(out)
    out = Dense(classes, activation='softmax')(out)

    model = Model(inp, out)

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model



