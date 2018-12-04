from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import tensorflow as tf
from keras.backend import tensorflow_backend

from scipy import misc
import os
import cv2
import numpy as np

np.random.seed(0)
np.random.RandomState(0)
tf.set_random_seed(0)


class DCGAN():
    def __init__(self, input_width, input_height, channel, output_width, output_height, z_dim):
        self.input_width = input_width
        self.input_height = input_height
        self.channel = channel
        self.output_width = output_width
        self.output_height = output_height

        self.input_shape = (self.input_width, self.input_width, channel)
        self.z_dim = z_dim

        optimizer = Adam(lr=0.0002, beta_1=0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.generator = self.build_generator()

        z = Input(shape=(self.z_dim,))
        gen_img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(gen_img)

        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        noise_shape = (self.z_dim,)

        model = Sequential()

        model.add(Dense(512 * self.output_width/8 * self.output_height/8, input_shape=noise_shape))
        model.add(Reshape((self.output_width/8, self.output_height/8, 512)))
        model.add(Activation("relu"))

        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(Activation("relu"))

        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))

        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))

        model.add(Conv2D(self.channel, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        gen_img = model(noise)

        return Model(noise, gen_img)

    def build_discriminator(self):
        img_shape = self.input_shape

        model = Sequential()

        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=img_shape)
        valid = model(img)

        return Model(img, valid)

    def build_combined(self):
        self.discriminator.trainable = False
        model = Sequential([self.generator, self.discriminator])

        return model

    def train(self, real_iamges, iterations, batch_size=128, save_interval=50, check_noise=None, n=None):

        half_batch = int(batch_size/2)
        real_iamges = (real_iamges.astype(np.float32) - 127.5) / 127.5

        for iteration in range(iterations):

            # ------------------
            # Training Discriminator
            # -----------------
            idx = np.random.randint(0, real_iamges.shape[0], half_batch)

            train_real_images = real_iamges[idx]

            noise = np.random.uniform(-1, 1, (half_batch, self.z_dim))

            gen_images = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(train_real_images, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_images, np.zeros((half_batch, 1)))

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -----------------
            # Training Generator
            # -----------------

            noise = np.random.uniform(-1, 1, (batch_size, self.z_dim))
            g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (iteration, d_loss[0], 100 * d_loss[1], g_loss))

            if iteration % save_interval == 0:
                self.save_imgs(iteration, check_noise, n)

        self.generator.save('generator.hdf5')
        self.discriminator.save('discriminator.hdf5')

    def test(self, number):
        test_noise = np.random.uniform(-1, 1, (number, 100))
        gen_imgs = self.generator.predict(test_noise)
        for i in range(number):
            misc.imsave('test/' + str(i) + '.png', gen_imgs[i])
        return

    def load_imgs(self, file_path):
        images = []
        imgs = os.listdir(file_path)
        for img in imgs:
            img_names = os.path.join(file_path, img)
            img = cv2.imread(img_names)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(img, dtype='float32')
            img = cv2.resize(img, (64, 64))
            images.append(img)

        images = np.array(images)
        return images

    def save_imgs(self, iteration, check_noise, n):
        gen_imgs = self.generator.predict(check_noise)
        for i in range(n):
            misc.imsave('train/' + str(i) + str(iteration) + '.png', gen_imgs[i])
        return


if __name__ == '__main__':
    root_dir = "flower_images"  # file path of training dataset
    input_height = 64  # the height of the input image
    input_width = 64   # the width of the input image
    input_channel = 3  # the channel of the input image
    output_height = 64 # the height of the output image
    output_width = 64  # the height of the output image
    z_dim = 100        # the dimension of the noise z
    dcgan = DCGAN(input_height, input_width, input_channel, output_height, output_width, z_dim)
    if not os.path.exists('train'):
        os.makedirs('train')
    if not os.path.exists('test'):
        os.makedirs('test')
    n = 16
    check_noise = np.random.uniform(-1, 1, (n, 100))
    images = dcgan.load_imgs(root_dir)

    dcgan.train(images, iterations=200, batch_size=64, save_interval=10, check_noise=check_noise, n=n)
    dcgan.test(100)