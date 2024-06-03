import tensorflow as tf

import sys

sys.path.append("./") # Append directory where python is invoked
from config import SEED

from tensorflow.keras.layers import Conv2D, Dropout, Dense, BatchNormalization, MaxPool2D, Flatten
from tensorflow.keras.initializers import HeNormal

from tensorflow.keras.models import load_model

def contrastive_loss(y_true, y_pred):    
    margin = 1
    loss = (1 - y_true) * tf.square(y_pred) +  y_true * tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(loss)

def instanciate_siamese_conv():
    siamese_conv = tf.keras.models.Sequential()
    siamese_conv.add(Conv2D(filters=16,
                            kernel_size=(2, 2),
                            kernel_initializer=HeNormal(seed=SEED),
                            activation='relu', padding="same"))
    siamese_conv.add(BatchNormalization())
    siamese_conv.add(Dropout(0.1))
    siamese_conv.add(Conv2D(filters=16,
                            kernel_size=(2, 2),
                            kernel_initializer=HeNormal(seed=SEED),
                            activation='relu', padding="same"))
    siamese_conv.add(BatchNormalization())
    siamese_conv.add(MaxPool2D((2, 2)))
    siamese_conv.add(Dropout(0.1))
    siamese_conv.add(Flatten())
    siamese_conv.add(Dense(
                        units=64,
                        activation="relu", kernel_initializer=HeNormal(seed=SEED)))
    siamese_conv.add(Dropout(0.1))
    siamese_conv.add(Dense(
                        units=32,
                        activation="relu", kernel_initializer=HeNormal(seed=SEED)))
    siamese_conv.add(Dropout(0.1))
    siamese_conv.add(Dense(
                        units=64,
                        activation="relu", kernel_initializer=HeNormal(seed=SEED)))
    siamese_conv.add(Dropout(0.1))
    siamese_conv.add(Dense(
                        units=128,
                        activation="relu", kernel_initializer=HeNormal(seed=SEED)))
    siamese_conv.add(Dropout(0.1))

    siamese_conv.load_weights('./test/models/siamese_best_weights.weights.h5')
    siamese_conv.compile(optimizer='rmsprop', loss=contrastive_loss, metrics=['accuracy'])

    return siamese_conv



