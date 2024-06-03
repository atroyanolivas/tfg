import sys

sys.path.append("../")
from config import INPUT_SHAPE, SEED

import tensorflow as tf
from functools import partial
from tensorflow.keras.layers import Conv2D, Dropout

from tensorflow.keras import regularizers
from tensorflow.keras.initializers import GlorotNormal, HeNormal

l2_rate = 0.002

# best_model = tf.keras.Sequential([
#   Conv2D(filters=64, kernel_size=(2, 2), 
#         kernel_initializer=HeNormal(seed=SEED),
#          #   kernel_regularizer=regularizers.l2(0.1), 
#            activation='relu', padding="same", input_shape=INPUT_SHAPE),
#   tf.keras.layers.BatchNormalization(),
#   Conv2D(filters=64, kernel_size=(2, 2), 
#         kernel_initializer=HeNormal(seed=SEED),
#          #   kernel_regularizer=regularizers.l2(0.1), 
#          activation='relu', padding="same"),
#   tf.keras.layers.BatchNormalization(),
#   tf.keras.layers.MaxPool2D((2, 2)),
#   Dropout(0.1),
#   Conv2D(filters=32, kernel_size=(2, 2), 
#         kernel_initializer=HeNormal(seed=SEED),
#          #   kernel_regularizer=regularizers.l2(0.1), 
#          activation='relu', padding="same"),
#   tf.keras.layers.BatchNormalization(),
#   Conv2D(filters=32, kernel_size=(2, 2), 
#         kernel_initializer=HeNormal(seed=SEED),
#          #   kernel_regularizer=regularizers.l2(0.1), 
#          activation='relu', padding="same"),
#   tf.keras.layers.BatchNormalization(),
#   tf.keras.layers.MaxPool2D((2, 2)),
#   Dropout(0.1),
#   Conv2D(filters=16, kernel_size=(2, 2), 
#         kernel_initializer=HeNormal(seed=SEED),
#          #   kernel_regularizer=regularizers.l2(0.1), 
#          activation='relu', padding="same"),
#   tf.keras.layers.BatchNormalization(),
#   Conv2D(filters=16, kernel_size=(2, 2), 
#         kernel_initializer=HeNormal(seed=SEED),
#          #   kernel_regularizer=regularizers.l2(0.1), 
#          activation='relu', padding="same"),
#   tf.keras.layers.BatchNormalization(),
#   tf.keras.layers.MaxPool2D((2, 2)),
#   Dropout(0.1),
#   Conv2D(filters=8, kernel_size=(2, 2), 
#         kernel_initializer=HeNormal(seed=SEED),
#          #   kernel_regularizer=regularizers.l2(0.1), 
#          activation='relu', padding="same"),
#   tf.keras.layers.BatchNormalization(),
#   Conv2D(filters=8, kernel_size=(2, 2), 
#         kernel_initializer=HeNormal(seed=SEED),
#          #   kernel_regularizer=regularizers.l2(0.1), 
#          activation='relu', padding="same"),
   
#   tf.keras.layers.BatchNormalization(),
#   tf.keras.layers.MaxPool2D((2, 2)),
#   Dropout(0.1),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(64,
#                           kernel_initializer=HeNormal(seed=SEED),
#                         #   kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#   Dropout(0.2),
#   tf.keras.layers.Dense(64,
#                           kernel_initializer=HeNormal(seed=SEED),
#                         #   kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#   Dropout(0.2),
#   tf.keras.layers.Dense(128,
#                           kernel_initializer=HeNormal(seed=SEED),
#                         #   kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#   #  Dropout(0.2),
#   #  tf.keras.layers.Dense(64,
#   #                         kernel_initializer=HeNormal(seed=SEED),
#   #                       #   kernel_regularizer=regularizers.l2(l2_rate), 
#   #                         activation="relu"),
#    tf.keras.layers.Dense(3, activation="softmax")
# ])

# BEST MODEL - with (32, 32) sized images

model_shorter = tf.keras.Sequential([
  Conv2D(filters=16, kernel_size=(2, 2), # 16
        kernel_initializer=HeNormal(seed=SEED),
         #   kernel_regularizer=regularizers.l2(0.1), 
           activation='relu', padding="same", input_shape=INPUT_SHAPE),
  tf.keras.layers.BatchNormalization(),
  Conv2D(filters=16, kernel_size=(2, 2), # 16
        kernel_initializer=HeNormal(seed=SEED),
         #   kernel_regularizer=regularizers.l2(0.1), 
         activation='relu', padding="same"),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPool2D((2, 2)),
  Dropout(0.1),
  Conv2D(filters=8, kernel_size=(2, 2), # 8
        kernel_initializer=HeNormal(seed=SEED),
         #   kernel_regularizer=regularizers.l2(0.1), 
         activation='relu', padding="same"),
  tf.keras.layers.BatchNormalization(),
  Conv2D(filters=8, kernel_size=(2, 2), # 8
        kernel_initializer=HeNormal(seed=SEED),
         #   kernel_regularizer=regularizers.l2(0.1), 
         activation='relu', padding="same"),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPool2D((2, 2)),
  Dropout(0.1),
  Conv2D(filters=4, kernel_size=(2, 2), # 4
        kernel_initializer=HeNormal(seed=SEED),
         #   kernel_regularizer=regularizers.l2(0.1), 
         activation='relu', padding="same"),
  tf.keras.layers.BatchNormalization(),
  Conv2D(filters=4, kernel_size=(2, 2), # 4
        kernel_initializer=HeNormal(seed=SEED),
         #   kernel_regularizer=regularizers.l2(0.1), 
         activation='relu', padding="same"),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPool2D((2, 2)),
  Dropout(0.1),
  Conv2D(filters=2, kernel_size=(2, 2), # 2
        kernel_initializer=HeNormal(seed=SEED),
         #   kernel_regularizer=regularizers.l2(0.1), 
         activation='relu', padding="same"),
  tf.keras.layers.BatchNormalization(),
  Conv2D(filters=2, kernel_size=(2, 2), # 2
        kernel_initializer=HeNormal(seed=SEED),
         #   kernel_regularizer=regularizers.l2(0.1), 
         activation='relu', padding="same"),
         
#       tf.keras.layers.BatchNormalization(),
#   tf.keras.layers.MaxPool2D((2, 2)),
   
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPool2D((2, 2)),
  Dropout(0.1),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512,
                          kernel_initializer=HeNormal(seed=SEED),
                        #   kernel_regularizer=regularizers.l2(l2_rate), 
                          activation="relu"),
  Dropout(0.2),
  tf.keras.layers.Dense(32, # 32
                          kernel_initializer=HeNormal(seed=SEED),
                        #   kernel_regularizer=regularizers.l2(l2_rate), 
                          activation="relu"),
  #  Dropout(0.2),
  #  tf.keras.layers.Dense(64,
  #                         kernel_initializer=HeNormal(seed=SEED),
  #                       #   kernel_regularizer=regularizers.l2(l2_rate), 
  #                         activation="relu"),
   tf.keras.layers.Dense(3, activation="softmax")
])
