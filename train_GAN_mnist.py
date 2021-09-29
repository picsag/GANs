import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Flatten, LeakyReLU, Dense, BatchNormalization, \
    Reshape, Conv2DTranspose, Dropout

import numpy as np
import time

config = {
    'BATCH_SIZE': 250,
}

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

plt.imshow(train_images[0], cmap=plt.get_cmap('gray'))
plt.show()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)

train_images = (train_images-127.5)/127.5

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).\
    shuffle(buffer_size=train_images.shape[0]).batch(config['BATCH_SIZE'])


def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))

    return model


def get_discriminator_loss(real_predictions, fake_predictions):
    # real_predictions = tf.sigmoid(real_predictions)
    # fake_predictions = tf.sigmoid(fake_predictions)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = cross_entropy(tf.ones_like(real_predictions), real_predictions)
    fake_loss = cross_entropy(tf.zeros_like(fake_predictions), fake_predictions)
    total_loss = real_loss + fake_loss
    return total_loss


# def get_discriminator_loss(real_output, generated_output):
#     # [1,1,...,1] with real output since it is true and we want our generated examples to look like it
#     real_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_output), logits=real_output)
#
#     # [0,0,...,0] with generated images since they are fake
#     generated_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(generated_output), logits=generated_output)
#
#     total_loss = real_loss + generated_loss
#
#     return total_loss

def generator_model():
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


def get_generator_loss(fake_predictions):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_predictions), fake_predictions)

    # fake_predictions = tf.sigmoid(fake_predictions)
    # fake_loss = tf.losses.binary_crossentropy(tf.zeros_like(fake_predictions), fake_predictions)
    # return fake_loss


def train_step(batch_size, images, generator_model: Model, discriminator_model: Model):
    """
    A train step will take only images
    First there is some fake noise, to be put into the generator to make some fake images.
    With these fake images, a real output and a fake output will be produced,
    i.e. what is the chance of this image being real or fake?
    """
    fake_image_noise = np.random.randn(batch_size, 100).astype('float32')

    discriminator_optimizer = tf.optimizers.Adam(1e-4)
    generator_optimizer = tf.optimizers.Adam(1e-4)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator_model(fake_image_noise, training=True)
        real_output = model_discriminator(images, training=True)
        fake_output = model_discriminator(generated_images)

        gen_loss = get_generator_loss(fake_output)
        disc_loss = get_discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator_model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, model_discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_model.trainable_variables))

    print(f"Generator loss: {np.mean(gen_loss)} -- Discriminator loss: {np.mean(disc_loss)}")


def train(dataset, epochs, batch_size, generator_model: Model, discriminator_model: Model):
    for _ in range(epochs):
        for images in dataset:
            images = tf.cast(images, tf.dtypes.float32)
            train_step(batch_size, images, generator_model, discriminator_model)



if __name__ == '__main__':
    model_discriminator = discriminator_model()

    model_discriminator(np.random.rand(1, 28, 28, 1).astype("float32"))

    model_generator = generator_model()

    train(train_dataset, epochs=30, batch_size=config['BATCH_SIZE'], generator_model=model_generator, discriminator_model=model_discriminator)

    plt.imshow(tf.reshape(model_generator(np.random.randn(1, 100)), (28, 28)), cmap="gray")

    plt.show()

    plt.imshow(tf.reshape(model_generator(np.random.randn(1, 100)), (28, 28)), cmap="gray")

    plt.show()





