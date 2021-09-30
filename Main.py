from typing import Dict
import tensorflow as tf
from configuration import configuration
from GANBuilder import GANBuilder


def generate_dataset(config: Dict):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # plt.imshow(train_images[0], cmap=plt.get_cmap('gray'))
    # plt.show()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)

    train_images = (train_images-127.5)/127.5

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).\
        shuffle(buffer_size=train_images.shape[0]).batch(config['BATCH_SIZE'])

    return train_dataset


if __name__ == '__main__':
    dataset = generate_dataset(config=configuration)

    GAN_net = GANBuilder(config=configuration, use_case=configuration['USE_CASES'][0], dataset=dataset)

    GAN_net.train_GAN()

