import os
import logging
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, Flatten, LeakyReLU, Dense, BatchNormalization, \
    Reshape, Conv2DTranspose, Dropout

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class GANBuilder(object):
    def __init__(self, config, use_case, dataset) -> None:
        super().__init__()
        self.config = config
        self.epochs = config['EPOCHS']
        self.batch_size = config['BATCH_SIZE']
        self.dataset = dataset
        self.use_case = use_case
        self.random_input_size = config['RANDOM_INPUT_SIZE']
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        if self.use_case == 'MNIST':
            self.discriminator_model = self.build_discriminator_model_MNIST()
            self.generator_model = self.build_generator_model_MNIST()
        elif self.use_case == 'CIFAR10':
            self.discriminator_model = self.build_discriminator_model_CIFAR10()
            self.generator_model = self.build_generator_model_CIFAR10()
        else:
            logging.log('ERROR', f"Could not load settings for use case: {self.use_case}")

        self.checkpoint_filepath = 'C:/WORK/Projects/GANs/checkpoint_' + self.use_case + '/input_size_' + \
                                   str(self.random_input_size)

    def get_generator_model(self):
        try:
            model = load_model(self.checkpoint_filepath)
            return model
        except OSError:
            print(f"Could not load model from: {self.checkpoint_filepath}")
            logging.log(logging.ERROR, f"Could not load model from: {self.checkpoint_filepath}")
            return None

    def build_discriminator_model_MNIST(self, dropout=0.3):
        model = Sequential()
        model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                         input_shape=[28, 28, 1]))
        model.add(LeakyReLU())
        model.add(Dropout(dropout))

        model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(dropout))

        model.add(Flatten())
        model.add(Dense(1))

        return model

    def build_discriminator_model_CIFAR10(self, dropout=0.4, in_shape=(32, 32, 3)):
        model = Sequential()
        # normal
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=in_shape))
        model.add(LeakyReLU(alpha=0.2))
        # downsample
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # downsample
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # downsample
        model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # classifier
        model.add(Flatten())
        model.add(Dropout(dropout))
        model.add(Dense(1, activation='sigmoid'))
        # compile model
        opt = tf.optimizers.Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def build_generator_model_MNIST(self):
        model = Sequential()
        model.add(Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(BatchNormalization(trainable=False))
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)

        return model

    def build_generator_model_CIFAR10(self):
        model = Sequential()
        # foundation for 4x4 image
        n_nodes = 256 * 4 * 4
        model.add(Dense(n_nodes, input_dim=self.random_input_size))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((4, 4, 256)))
        # upsample to 8x8
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 16x16
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 32x32
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # output layer
        model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
        return model

    def get_discriminator_loss(self, real_predictions, fake_predictions):
        real_loss = self.cross_entropy(tf.ones_like(real_predictions), real_predictions)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_predictions), fake_predictions)
        total_loss = real_loss + fake_loss
        return total_loss

    def get_generator_loss(self, fake_predictions):
        return self.cross_entropy(tf.ones_like(fake_predictions), fake_predictions)

    def train_step(self, images):
        fake_image_noise = np.random.randn(self.batch_size, self.random_input_size).astype('float32')

        discriminator_optimizer = tf.optimizers.Adam(1e-4)
        generator_optimizer = tf.optimizers.Adam(1e-4)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator_model(fake_image_noise, training=True)
            real_output = self.discriminator_model(images, training=True)
            fake_output = self.discriminator_model(generated_images)

            gen_loss = self.get_generator_loss(fake_output)
            disc_loss = self.get_discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator_model.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator_model.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator_model.trainable_variables))
        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator_model.trainable_variables))

        print(f"Generator loss: {np.mean(gen_loss)} -- Discriminator loss: {np.mean(disc_loss)}")

    def train_GAN(self):

        # self.discriminator_model(np.random.rand(1, 28, 28, 1).astype("float32"))

        for _ in range(self.config['EPOCHS']):
            for images in self.dataset:
                images = tf.cast(images, tf.dtypes.float32)
                self.train_step(images)

    def generate_images(self, n_samples):
        # generate points in latent space as input for the generator
        # generate points in the latent space
        x_input = np.randn(self.random_input_size * n_samples)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n_samples, self.random_input_size)
        predictions = self.generator_model(x_input, training=False)

        for i in range(predictions.shape[0]):
            # img = predictions[i, :, :, :] * 127.5 + 127.5
            # matplotlib.image.imsave('name.png', )
            plt.imshow(predictions[i, :, :, :] * 127.5 + 127.5)
            plt.axis('off')
            plt.savefig(f"predicted_image_{i}.png")

        return predictions