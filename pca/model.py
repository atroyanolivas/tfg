import sys

sys.path.append("./")

from config import N_COMP, INPUT_SHAPE, SEED, PCA_LEARN_RATE

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import HeNormal

from tensorflow.keras.optimizers import Adam


dropout_rate = 0.2
l2_rate = 0.1
model = tf.keras.Sequential([
    Dense(1024, activation="relu",
        kernel_initializer=HeNormal(seed=SEED),
        # kernel_regularizer=regularizers.l2(l2_rate)
        ),
    # tf.keras.layers.BatchNormalization(),
    # Dropout(dropout_rate),
    Dense(128, activation="relu",
        kernel_initializer=HeNormal(seed=SEED),
        # kernel_regularizer=regularizers.l2(l2_rate)
        ),
    # tf.keras.layers.BatchNormalization(),
    # Dropout(dropout_rate),
    Dense(32, activation="relu",
        kernel_initializer=HeNormal(seed=SEED),
        # kernel_regularizer=regularizers.l2(l2_rate)
        ),
    # tf.keras.layers.BatchNormalization(),

    Dense(3, activation="softmax"),
])

def build_pca_model(hp):

    model = tf.keras.models.Sequential()
    # model.add(Dense(
    #     units=hp.Int('units1', min_value=32, max_value=2048, step=32),
    #     activation="relu", kernel_initializer=HeNormal(seed=SEED)
    # ))
    # model.add(Dense(
    #     units=hp.Int('units2', min_value=32, max_value=1024, step=32),
    #     activation="relu", kernel_initializer=HeNormal(seed=SEED)
    # ))
    # model.add(Dense(
    #     units=hp.Int('units3', min_value=32, max_value=1024, step=32),
    #     activation="relu", kernel_initializer=HeNormal(seed=SEED)
    # ))  
    num_layers = hp.Int('num_layers', min_value=1, max_value=10, step=1)
    # dropout = hp.Choice('dropout?', values=[True, False])
    drop_rate = 0.1

    for i in range(1, num_layers+1):
        model.add(Dense(
            units=hp.Choice(f'units{i}', values=[32, 64, 96]),
            activation="relu", kernel_initializer=HeNormal(seed=SEED)
        ))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1, activation="sigmoid", kernel_initializer=HeNormal(seed=SEED)))
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice("learning_rate", values=[0.001, 0.01])),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

