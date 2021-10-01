import os
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model


class CNNBuilder(object):
    def __init__(self, config, coin, X_train, X_test, y_train, y_test) -> None:
        super().__init__()
        self.config = config
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.coin = coin
        self.epochs = config['EPOCHS']
        self.batch_size = config['BATCH_SIZE']
        self.checkpoint_filepath = 'C:/WORK/Projects/GANs/checkpoint_' + self.use_case + '/input_size_' + \
                                   str(self.random_input_size) + '_CNN'

    def get_model(self, model_type: str):
        try:
            model = load_model(os.path.join(self.checkpoint_filepath, model_type))
            return model
        except OSError:
            print(f"Could not load model from: {os.path.join(self.checkpoint_filepath, model_type)}")
            logging.log(logging.ERROR, "Cannot predict price")
            return None

    def train_model(self, model_type: str):
        if model_type == "CNN1":
            self.checkpoint_filepath += '/CNN1'
            return self.train_CNN1_model()
        elif model_type == "bi_lstm":
            self.checkpoint_filepath += '/CNN2'
            return self.train_CNN2_model()

    def build_CNN1_model(self, no_classes=10, optimizer='adam'):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(no_classes))

        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        return model

    def train_CNN1_model(self, no_classes=10, optimizer='adam', epochs=70, batch_size=50):
        model = self.build_CNN1_model(no_classes=no_classes, optimizer=optimizer)

        model_checkpoint_callback = ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        history = model.fit(
            self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True,
            validation_split=0.2, callbacks=[model_checkpoint_callback]
        )

        return model

    def build_CNN2_model(self, dropout=0.5, no_classes=10, optimizer='adam'):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
        model.add(Flatten())
        model.add(Dropout(dropout, noise_shape=None, seed=None))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(no_classes))

        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        return model

    def train_CNN2_model(self, no_classes=10, optimizer='adam', epochs=70, batch_size=50):
        model = self.build_CNN2_model(no_classes=no_classes, optimizer=optimizer)

        model_checkpoint_callback = ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        history = model.fit(
            self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True,
            validation_split=0.2, callbacks=[model_checkpoint_callback]
        )

        return model




