"""



"""
import sys

sys.path.append("../")
from config import INPUT_SHAPE, SEED

import tensorflow as tf
from functools import partial
from tensorflow.keras.layers import Conv2D, Dropout, Dense

from tensorflow.keras import regularizers
from tensorflow.keras.initializers import GlorotNormal, HeNormal
from tensorflow.keras.optimizers import Adam

N = 50
M = 15


model_vgg8 = tf.keras.Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding="same",
                        #   kernel_initializer="he_normal", 
                          input_shape=INPUT_SHAPE),
    tf.keras.layers.MaxPool2D((2, 2)),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"),
                        #   kernel_initializer="he_normal"),
    tf.keras.layers.MaxPool2D((2, 2)),
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding="same"),
                        #   kernel_initializer="he_normal"),
    tf.keras.layers.MaxPool2D((2, 2)),
    Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding="same"),
                        #   kernel_initializer="he_normal"),
    tf.keras.layers.MaxPool2D((2, 2)),
    Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding="same"),
                        #   kernel_initializer="he_normal"),
    tf.keras.layers.MaxPool2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(N, activation="relu"),
    tf.keras.layers.Dense(M, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

model_short = tf.keras.Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding="same", input_shape=INPUT_SHAPE), 
    tf.keras.layers.MaxPool2D((2, 2)),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"), 
    tf.keras.layers.MaxPool2D((2, 2)),
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding="same"), 
    tf.keras.layers.MaxPool2D((2, 2)),
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding="same"), 
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(N, activation="relu"),
    tf.keras.layers.Dense(M, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax") # 2 neurons: probability of each class
])

dropout_rate = 0.07
l2_rate = 0.0025

# model_shorter = tf.keras.Sequential([
#     Conv2D(filters=32, kernel_size=(7, 7), 
#            kernel_initializer=GlorotNormal(seed=None),
#         #    kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same", input_shape=INPUT_SHAPE), 
#     Dropout(dropout_rate),
#     tf.keras.layers.MaxPool2D((2, 2)),
#     Conv2D(filters=32, kernel_size=(5, 5), 
#            kernel_initializer=GlorotNormal(seed=None), 
#         #    kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same"), 
#     Dropout(dropout_rate),
#     tf.keras.layers.MaxPool2D((2, 2)),
#     Conv2D(filters=64, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None), 
#         #    kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same"), 
#     Dropout(dropout_rate),
#     tf.keras.layers.MaxPool2D((2, 2)),
#     Conv2D(filters=64, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None), 
#         #    kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same"), 
#     Dropout(dropout_rate),
#     tf.keras.layers.MaxPool2D((2, 2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(20, 
#                           kernel_initializer=GlorotNormal(seed=None),
#                         #   kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     Dropout(dropout_rate),
#     # tf.keras.layers.Dense(10, activation="relu"),
#     tf.keras.layers.Dense(3, activation="softmax") # 2 neurons: probability of each class
# ])
dropout_rate = 0.05
l2_rate = 0.0025

# model_shorter = tf.keras.Sequential([
#     Conv2D(filters=32, kernel_size=(5, 5), 
#            kernel_initializer=GlorotNormal(seed=None),
#         #    kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same", input_shape=INPUT_SHAPE), 
#     Dropout(dropout_rate),
#     # tf.keras.layers.MaxPool2D((2, 2)),
#     # Conv2D(filters=16, kernel_size=(5, 5), 
#     #        kernel_initializer=GlorotNormal(seed=None), 
#     #     #    kernel_regularizer=regularizers.l2(l2_rate), 
#     #        activation='relu', padding="same"), 
#     # Dropout(dropout_rate),
#     tf.keras.layers.MaxPool2D((2, 2)),
#     Conv2D(filters=32, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None), 
#         #    kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same"), 
#     Dropout(dropout_rate),
#     tf.keras.layers.MaxPool2D((2, 2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(25, 
#                           kernel_initializer=GlorotNormal(seed=None),
#                         #   kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     # Dropout(dropout_rate),
#     tf.keras.layers.Dense(10, 
#                           kernel_initializer=GlorotNormal(seed=None),
#                         #   kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     # Dropout(dropout_rate),
#     tf.keras.layers.Dense(10, 
#                           kernel_initializer=GlorotNormal(seed=None),
#                         #   kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     # Dropout(dropout_rate),
#     tf.keras.layers.Dense(3, activation="softmax") # 2 neurons: probability of each class
# ])

# dropout_rate = 0.05
# dropout_rate = 0.065
# l2_rate = 0.0025

# model_shorter = tf.keras.Sequential([
#     Conv2D(filters=32, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None),
#           #  kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same", input_shape=INPUT_SHAPE), 
#     # Dropout(dropout_rate),
#     tf.keras.layers.MaxPool2D((2, 2)),
#     Conv2D(filters=64, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None),
#           #  kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same"), 
#     # Dropout(dropout_rate),
#     tf.keras.layers.MaxPool2D((2, 2)),
#     Conv2D(filters=64, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None), 
#           #  kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same"), 
#     Dropout(dropout_rate),
#     tf.keras.layers.MaxPool2D((2, 2)),
#     Conv2D(filters=128, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None), 
#           #  kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same"), 
#     # Dropout(dropout_rate),
#     tf.keras.layers.MaxPool2D((2, 2)),
#     Conv2D(filters=128, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None), 
#           #  kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same"),
#     # Dropout(dropout_rate),
#     tf.keras.layers.MaxPool2D((2, 2)),
#     Conv2D(filters=256, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None), 
#           #  kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same"),
#     # Dropout(dropout_rate),
#     tf.keras.layers.MaxPool2D((2, 2)),
#     Conv2D(filters=256, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None), 
#           #  kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same"),
#     Dropout(dropout_rate),
#     tf.keras.layers.MaxPool2D((2, 2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512, #700?
#                           kernel_initializer=GlorotNormal(seed=None),
#                           # kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     Dropout(dropout_rate),
#     tf.keras.layers.Dense(512, 
#                           kernel_initializer=GlorotNormal(seed=None),
#                         #   kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     Dropout(dropout_rate),
#     tf.keras.layers.Dense(256, 
#                           kernel_initializer=GlorotNormal(seed=None),
#                         #   kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     Dropout(dropout_rate),
#     tf.keras.layers.Dense(256, 
#                           kernel_initializer=GlorotNormal(seed=None),
#                         #   kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     Dropout(dropout_rate),
#     tf.keras.layers.Dense(128, 
#                           kernel_initializer=GlorotNormal(seed=None),
#                         #   kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     Dropout(dropout_rate),
#     tf.keras.layers.Dense(64, 
#                           kernel_initializer=GlorotNormal(seed=None),
#                         #   kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     Dropout(dropout_rate),
#     tf.keras.layers.Dense(3, activation="softmax") # 2 neurons: probability of each class
# ])




# dropout_rate = 0.05
# # dropout_rate = 0.065
# l2_rate = 0.0025

# model_shorter = tf.keras.Sequential([
#     Conv2D(filters=128, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None),
#           #  kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same", input_shape=INPUT_SHAPE), 
# #     Dropout(dropout_rate),
#     # tf.keras.layers.MaxPool2D((2, 2)),
#     Conv2D(filters=128, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None),
#           #  kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same"), 
# #     Dropout(0.02),
#     tf.keras.layers.MaxPool2D((2, 2)),
#     Conv2D(filters=64, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None), 
#           #  kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same"), 
# #     Dropout(0.02),
#     # tf.keras.layers.MaxPool2D((2, 2)),
#     Conv2D(filters=64, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None), 
#           #  kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same"), 
# #     Dropout(0.02),
#     tf.keras.layers.MaxPool2D((2, 2)),
#     Conv2D(filters=32, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None), 
#           #  kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same"),
# #     Dropout(0.02),
#     tf.keras.layers.MaxPool2D((2, 2)),
#     Conv2D(filters=32, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None), 
#           #  kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same"),
#     Dropout(dropout_rate),
#     tf.keras.layers.MaxPool2D((2, 2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512, #768?
#                           kernel_initializer=GlorotNormal(seed=None),
#                           # kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     Dropout(dropout_rate),
#     tf.keras.layers.Dense(1024, 
#                           kernel_initializer=GlorotNormal(seed=None),
#                         #   kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     Dropout(dropout_rate),
#     tf.keras.layers.Dense(1024, 
#                           kernel_initializer=GlorotNormal(seed=None),
#                         #   kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     Dropout(dropout_rate),
#     tf.keras.layers.Dense(512, 
#                           kernel_initializer=GlorotNormal(seed=None),
#                         #   kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     Dropout(dropout_rate),
#     tf.keras.layers.Dense(256, 
#                           kernel_initializer=GlorotNormal(seed=None),
#                         #   kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     Dropout(dropout_rate),
#     tf.keras.layers.Dense(64, 
#                           kernel_initializer=GlorotNormal(seed=None),
#                         #   kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     Dropout(dropout_rate),
#     tf.keras.layers.Dense(3, activation="softmax") # 2 neurons: probability of each class
# ])

# dropout_rate = 0.05
# l2_rate = 0.0025

# model_shorter = tf.keras.Sequential([
#     Conv2D(filters=32, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None),
#            kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same", input_shape=INPUT_SHAPE), 
# #     Dropout(dropout_rate),
#     Conv2D(filters=32, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None),
#            kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same"), 
# #     Dropout(0.02),
#     tf.keras.layers.MaxPool2D((2, 2), strides=2),
#     Conv2D(filters=64, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None), 
#            kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same"), 
# #     Dropout(0.02),
#     Conv2D(filters=64, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None), 
#            kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same"), 
# #     Dropout(0.02),
#     tf.keras.layers.MaxPool2D((2, 2), strides=2),
#     Conv2D(filters=128, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None), 
#            kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same"),
# #     Dropout(0.02),
#     Conv2D(filters=128, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None), 
#            kernel_regularizer=regularizers.l2(l2_rate), 
#            activation='relu', padding="same"),
#     Dropout(dropout_rate),
#     tf.keras.layers.MaxPool2D((2, 2), strides=2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512, #768?
#                           kernel_initializer=GlorotNormal(seed=None),
#                           # kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     Dropout(dropout_rate),
#     tf.keras.layers.Dense(1024, 
#                           kernel_initializer=GlorotNormal(seed=None),
#                           # kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     # Dropout(dropout_rate),
#     # tf.keras.layers.Dense(1024, 
#     #                       kernel_initializer=GlorotNormal(seed=None),
#     #                     #   kernel_regularizer=regularizers.l2(l2_rate), 
#     #                       activation="relu"),
#     # Dropout(dropout_rate),
#     # tf.keras.layers.Dense(512, 
#     #                       kernel_initializer=GlorotNormal(seed=None),
#     #                     #   kernel_regularizer=regularizers.l2(l2_rate), 
#     #                       activation="relu"),
#     Dropout(dropout_rate),
#     tf.keras.layers.Dense(256, 
#                           kernel_initializer=GlorotNormal(seed=None),
#                           # kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     Dropout(dropout_rate),
#     tf.keras.layers.Dense(64, 
#                           kernel_initializer=GlorotNormal(seed=None),
#                           # kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     Dropout(dropout_rate),
#     tf.keras.layers.Dense(3, activation="softmax") # 2 neurons: probability of each class
# ])

# dropout_rate = 0.05
# l2_rate = 0.0025

# model_shorter = tf.keras.Sequential([
#     Conv2D(filters=32, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None),
#            kernel_regularizer=regularizers.l2(0.1), 
#            activation='relu', padding="same", input_shape=INPUT_SHAPE), 
#     Dropout(dropout_rate),
#     Conv2D(filters=32, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None),
#            kernel_regularizer=regularizers.l2(0.1), 
#            activation='relu', padding="same", input_shape=INPUT_SHAPE), 
#     Dropout(dropout_rate),
#     tf.keras.layers.MaxPool2D((2, 2), strides=2),
#     Conv2D(filters=64, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None),
#            kernel_regularizer=regularizers.l2(0.1), 
#            activation='relu', padding="same"),
#            Dropout(dropout_rate),
#     Conv2D(filters=64, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None),
#            kernel_regularizer=regularizers.l2(0.1), 
#            activation='relu', padding="same"),
#     Dropout(dropout_rate),
#     tf.keras.layers.MaxPool2D((2, 2), strides=2),
#     Conv2D(filters=128, kernel_size=(3, 3), 
#            kernel_initializer=GlorotNormal(seed=None),
#            kernel_regularizer=regularizers.l2(0.1) , 
#            activation='relu', padding="same"),
# #     Conv2D(filters=128, kernel_size=(3, 3), 
# #            kernel_initializer=GlorotNormal(seed=None),
# #           #  kernel_regularizer=regularizers.l2(l2_rate), 
# #            activation='relu', padding="same"),
#     Dropout(dropout_rate),
#     tf.keras.layers.MaxPool2D((2, 2), strides=2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(32, #768?
#                           kernel_initializer=GlorotNormal(seed=None),
#                           kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     Dropout(dropout_rate),
#     tf.keras.layers.Dense(64, 
#                           kernel_initializer=GlorotNormal(seed=None),
#                           kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     # Dropout(dropout_rate),
#     tf.keras.layers.Dense(3, activation="softmax") # 2 neurons: probability of each class
# ])

# 64x64
model_shorter = tf.keras.Sequential([
    # Conv2D(filters=64, kernel_size=(2, 2), 
    #     #    kernel_initializer=GlorotNormal(seed=None),
    #     #    kernel_regularizer=regularizers.l2(0.1), 
    #        activation='relu', padding="same", input_shape=INPUT_SHAPE),
    # tf.keras.layers.BatchNormalization(),
    # Conv2D(filters=64, kernel_size=(2, 2), 
    #     #    kernel_initializer=GlorotNormal(seed=None),
    #     #    kernel_regularizer=regularizers.l2(0.1), 
    #        activation='relu', padding="same"),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.MaxPool2D((2, 2)),
    # Dropout(0.1), # 0.035
    # Conv2D(filters=32, kernel_size=(2, 2), 
    #     #    kernel_initializer=GlorotNormal(seed=None),
    #     #    kernel_regularizer=regularizers.l2(0.1), 
    #        activation='relu', padding="same"),
    # tf.keras.layers.BatchNormalization(),
    Conv2D(filters=32, kernel_size=(2, 2), # 32
           kernel_initializer=HeNormal(seed=None),
        #    kernel_regularizer=regularizers.l2(0.1), 
           activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2, 2)),
    Dropout(0.1), # 0.035
    Conv2D(filters=16, kernel_size=(2, 2), 
           kernel_initializer=HeNormal(seed=None),
        #    kernel_regularizer=regularizers.l2(0.1), 
           activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    Conv2D(filters=16, kernel_size=(2, 2), 
           kernel_initializer=HeNormal(seed=None),
        #    kernel_regularizer=regularizers.l2(0.1), 
           activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2, 2)),
    Dropout(0.1), # 0.035
    Conv2D(filters=8, kernel_size=(2, 2), 
           kernel_initializer=HeNormal(seed=None),
        #    kernel_regularizer=regularizers.l2(0.1), 
           activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    Conv2D(filters=8, kernel_size=(2, 2), 
           kernel_initializer=HeNormal(seed=None),
        #    kernel_regularizer=regularizers.l2(0.1), 
           activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2, 2)),
    Dropout(0.1), # 0.035
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,
                          kernel_initializer=HeNormal(seed=None),
                        #   kernel_regularizer=regularizers.l2(l2_rate), 
                          activation="relu"),
    Dropout(0.2),
    tf.keras.layers.Dense(128,
                          kernel_initializer=HeNormal(seed=None),
                        #   kernel_regularizer=regularizers.l2(l2_rate), 
                          activation="relu"),
    Dropout(0.2),
    tf.keras.layers.Dense(128,
                          kernel_initializer=HeNormal(seed=None),
                        #   kernel_regularizer=regularizers.l2(l2_rate), 
                          activation="relu"),
    Dropout(0.2),
    tf.keras.layers.Dense(3, activation="softmax")
])
def build_conv_model(hp):
   model = tf.keras.models.Sequential()
   pool_num_choice = hp.Choice("pool_kernel", values=[2, 3])
   # pool_num_choice = hp.Choice("pool_kernel1", values=[2, 3])
   pool_choice = (pool_num_choice, pool_num_choice)
   drop_conv = hp.Choice("drop_conv", values=[0.1, 0.2])
   # drop_conv = hp.Choice("drop_conv", values=[0.05, 0.1])
   drop_dense = hp.Choice("drop_dense", values=[0.1, 0.2])
   # drop_dense = hp.Choice("drop_conv", values=[0.05, 0.1])
   # 1st block
   kernel_choice = hp.Choice("kernel_size1", values=[2, 3, 5, 7])
   # kernel_choice = hp.Choice("kernel_size1", values=[2])
   model.add(
      Conv2D(filters=hp.Int('filters1', min_value=8, max_value=128, step=8),
      # Conv2D(filters=hp.Choice("filters1", values=[2, 4, 6, 8, 12, 16]),
         kernel_size=(kernel_choice, kernel_choice),
         kernel_initializer=HeNormal(seed=SEED),
        #    kernel_regularizer=regularizers.l2(0.1), 
         activation='relu', padding="same"))
   model.add(tf.keras.layers.BatchNormalization())
   kernel_choice = hp.Choice("kernel_size2", values=[2, 3, 5, 7])
   # kernel_choice = hp.Choice("kernel_size2", values=[2])
   model.add(
      Conv2D(filters=hp.Int('filters2', min_value=8, max_value=128, step=8),
      # Conv2D(filters=hp.Choice("filters2", values=[2, 4, 6, 8, 12, 16]),
         kernel_size=(kernel_choice, kernel_choice),
         kernel_initializer=HeNormal(seed=SEED),
        #    kernel_regularizer=regularizers.l2(0.1), 
         activation='relu', padding="same"))
   model.add(tf.keras.layers.BatchNormalization())
   model.add(tf.keras.layers.MaxPool2D(pool_choice))
   model.add(Dropout(drop_conv))
   # 2nd block
   kernel_choice = hp.Choice("kernel_size3", values=[2, 3, 5, 7])
   # kernel_choice = hp.Choice("kernel_size3", values=[2])
   model.add(
      Conv2D(filters=hp.Int('filters3', min_value=8, max_value=128, step=8),
      # Conv2D(filters=hp.Choice("filters3", values=[2, 4, 6, 8, 12, 16]),
         kernel_size=(kernel_choice, kernel_choice),
         kernel_initializer=HeNormal(seed=SEED),
        #    kernel_regularizer=regularizers.l2(0.1), 
         activation='relu', padding="same"))
   model.add(tf.keras.layers.BatchNormalization())
   kernel_choice = hp.Choice("kernel_size4", values=[2, 3, 5, 7])
   # kernel_choice = hp.Choice("kernel_size4", values=[2])
   model.add(
      Conv2D(filters=hp.Int('filters4', min_value=8, max_value=128, step=8),
      # Conv2D(filters=hp.Choice("filters4", values=[2, 4, 6, 8, 12, 16]),
         kernel_size=(kernel_choice, kernel_choice),
         kernel_initializer=HeNormal(seed=SEED),
        #    kernel_regularizer=regularizers.l2(0.1), 
         activation='relu', padding="same"))
   model.add(tf.keras.layers.BatchNormalization())
   # pool_num_choice = hp.Choice("pool_kernel2", values=[2, 3])
   model.add(tf.keras.layers.MaxPool2D((pool_num_choice, pool_num_choice)))
   model.add(Dropout(drop_conv))
   # 3rd block
   # kernel_choice = hp.Choice("kernel_size5", values=[2, 3, 5, 7])
   # # kernel_choice = hp.Choice("kernel_size5", values=[2])
   # model.add(
   #    Conv2D(filters=hp.Int('filters5', min_value=8, max_value=128, step=8),
   #    # Conv2D(filters=hp.Choice("filters5", values=[2, 4, 6, 8, 12, 16]),
   #       kernel_size=(kernel_choice, kernel_choice),
   #       kernel_initializer=HeNormal(seed=SEED),
   #      #    kernel_regularizer=regularizers.l2(0.1), 
   #       activation='relu', padding="same"))
   # model.add(tf.keras.layers.BatchNormalization())
   # kernel_choice = hp.Choice("kernel_size6", values=[2, 3, 5, 7])
   # # kernel_choice = hp.Choice("kernel_size6", values=[2])
   # model.add(
   #    Conv2D(filters=hp.Int('filters6', min_value=8, max_value=128, step=8),
   #    # Conv2D(filters=hp.Choice("filters6", values=[2, 4, 6, 8, 12, 16]),
   #       kernel_size=(kernel_choice, kernel_choice),
   #       kernel_initializer=HeNormal(seed=SEED),
   #      #    kernel_regularizer=regularizers.l2(0.1), 
   #       activation='relu', padding="same"))
   # model.add(tf.keras.layers.BatchNormalization())
   # model.add(tf.keras.layers.MaxPool2D(pool_choice))
   # model.add(Dropout(drop_conv))
   model.add(tf.keras.layers.Flatten())
   model.add(Dense(
        units=hp.Int('units1', min_value=32, max_value=256, step=32),
         # units=hp.Choice('units1', values=[8, 16, 32, 48, 64]),
         activation="relu", kernel_initializer=HeNormal(seed=SEED)
   ))
   model.add(Dropout(drop_dense))
   model.add(Dense(
        units=hp.Int('units2', min_value=32, max_value=256, step=32),
         # units=hp.Choice('units2', values=[8, 16, 32, 64, 72, 96, 104]),
         activation="relu", kernel_initializer=HeNormal(seed=SEED)
   ))
   model.add(Dropout(drop_dense))
   model.add(Dense(
        units=hp.Int('units3', min_value=32, max_value=256, step=32),
         # units=hp.Choice('units3', values=[8, 16, 96, 104, 112, 120, 128]),
         activation="relu", kernel_initializer=HeNormal(seed=SEED)
   ))
   model.add(Dropout(drop_dense))
   model.add(Dense(1, activation="sigmoid"))

   model.compile(
      optimizer=Adam(learning_rate=hp.Choice("learning_rate", values=[0.01, 0.001, 0.0001, 0.00001])),
      # optimizer=Adam(learning_rate=hp.Choice("learning_rate", values=[0.01, 0.001])),
      loss="binary_crossentropy",
      metrics=["accuracy"]
   )

   return model



# 92x92
# model_shorter = tf.keras.Sequential([
#     # Conv2D(filters=64, kernel_size=(2, 2), 
#     #     #    kernel_initializer=GlorotNormal(seed=None),
#     #     #    kernel_regularizer=regularizers.l2(0.1), 
#     #        activation='relu', padding="same", input_shape=INPUT_SHAPE),
#     # tf.keras.layers.BatchNormalization(),
#     # Conv2D(filters=64, kernel_size=(2, 2), 
#     #     #    kernel_initializer=GlorotNormal(seed=None),
#     #     #    kernel_regularizer=regularizers.l2(0.1), 
#     #        activation='relu', padding="same"),
#     # tf.keras.layers.BatchNormalization(),
#     # tf.keras.layers.MaxPool2D((2, 2)),
#     # Dropout(0.1), # 0.035
#     Conv2D(filters=32, kernel_size=(2, 2), 
#         #    kernel_initializer=GlorotNormal(seed=None),
#         #    kernel_regularizer=regularizers.l2(0.1), 
#            activation='relu', padding="same"),
#     tf.keras.layers.BatchNormalization(),
#     Conv2D(filters=32, kernel_size=(2, 2), # 32
#            kernel_initializer=HeNormal(seed=None),
#         #    kernel_regularizer=regularizers.l2(0.1), 
#            activation='relu', padding="same"),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPool2D((2, 2)),
#     Dropout(0.1), # 0.035
#     Conv2D(filters=16, kernel_size=(2, 2), 
#            kernel_initializer=HeNormal(seed=None),
#         #    kernel_regularizer=regularizers.l2(0.1), 
#            activation='relu', padding="same"),
#     tf.keras.layers.BatchNormalization(),
#     Conv2D(filters=16, kernel_size=(2, 2), 
#            kernel_initializer=HeNormal(seed=None),
#         #    kernel_regularizer=regularizers.l2(0.1), 
#            activation='relu', padding="same"),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPool2D((2, 2)),
#     Dropout(0.1), # 0.035
#     Conv2D(filters=8, kernel_size=(2, 2), 
#            kernel_initializer=HeNormal(seed=None),
#         #    kernel_regularizer=regularizers.l2(0.1), 
#            activation='relu', padding="same"),
#    #  tf.keras.layers.BatchNormalization(),
#    #  Conv2D(filters=8, kernel_size=(2, 2), 
#    #         kernel_initializer=HeNormal(seed=None),
#    #      #    kernel_regularizer=regularizers.l2(0.1), 
#    #         activation='relu', padding="same"),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPool2D((2, 2)),
#     Dropout(0.1), # 0.035
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(1024,
#                           kernel_initializer=HeNormal(seed=None),
#                           kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     Dropout(0.2),
#     tf.keras.layers.Dense(256, # 128
#                           kernel_initializer=HeNormal(seed=None),
#                           kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     Dropout(0.2),
#     tf.keras.layers.Dense(128, # 128
#                           kernel_initializer=HeNormal(seed=None),
#                           kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#    #  Dropout(0.2),
#    #  tf.keras.layers.Dense(128,
#    #                        kernel_initializer=HeNormal(seed=None),
#    #                        kernel_regularizer=regularizers.l2(l2_rate), 
#    #                        activation="relu"),
#     Dropout(0.2),
#     tf.keras.layers.Dense(3, activation="softmax")
# ])

# l2_rate = 0.01

# model_shorter = tf.keras.Sequential([
#     Conv2D(filters=64, kernel_size=(2, 2), 
#         #    kernel_initializer=GlorotNormal(seed=None),
#         #    kernel_regularizer=regularizers.l2(0.1), 
#            activation='relu', padding="same", input_shape=INPUT_SHAPE),
#     tf.keras.layers.BatchNormalization(),
#     Conv2D(filters=64, kernel_size=(2, 2), 
#         #    kernel_initializer=GlorotNormal(seed=None),
#         #    kernel_regularizer=regularizers.l2(0.1), 
#            activation='relu', padding="same"),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPool2D((2, 2)),
#     Dropout(0.1), # 0.035
#     Conv2D(filters=32, kernel_size=(2, 2), 
#         #    kernel_initializer=GlorotNormal(seed=None),
#         #    kernel_regularizer=regularizers.l2(0.1), 
#            activation='relu', padding="same"),
#     tf.keras.layers.BatchNormalization(),
#     Conv2D(filters=32, kernel_size=(2, 2), # 32
#            kernel_initializer=HeNormal(seed=None),
#         #    kernel_regularizer=regularizers.l2(0.1), 
#            activation='relu', padding="same"),
#    tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPool2D((2, 2)),
#     Dropout(0.1), # 0.035
#     Conv2D(filters=16, kernel_size=(2, 2), 
#            kernel_initializer=HeNormal(seed=None),
#         #    kernel_regularizer=regularizers.l2(0.1), 
#            activation='relu', padding="same"),
#     tf.keras.layers.BatchNormalization(),
#     Conv2D(filters=16, kernel_size=(2, 2), 
#            kernel_initializer=HeNormal(seed=None),
#         #    kernel_regularizer=regularizers.l2(0.1), 
#            activation='relu', padding="same"),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPool2D((2, 2)),
#     Dropout(0.1), # 0.035
#     Conv2D(filters=8, kernel_size=(2, 2), 
#            kernel_initializer=HeNormal(seed=None),
#         #    kernel_regularizer=regularizers.l2(0.1), 
#            activation='relu', padding="same"),
#     tf.keras.layers.BatchNormalization(),
#     Conv2D(filters=8, kernel_size=(2, 2), 
#            kernel_initializer=HeNormal(seed=None),
#         #    kernel_regularizer=regularizers.l2(0.1), 
#            activation='relu', padding="same"),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPool2D((2, 2)),
#     Dropout(0.1), # 0.035
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(1024,
#                           kernel_initializer=HeNormal(seed=None),
#                           kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     Dropout(0.2),
#     tf.keras.layers.Dense(128,
#                           kernel_initializer=HeNormal(seed=None),
#                           kernel_regularizer=regularizers.l2(l2_rate), 
#                           activation="relu"),
#     Dropout(0.2),
#     tf.keras.layers.Dense(3, activation="softmax")
# ])


