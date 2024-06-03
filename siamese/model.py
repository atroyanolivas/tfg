"""


"""
import sys

sys.path.append("./") # Append directory where python is invoked

from config import INPUT_SHAPE, SEED
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import GlorotNormal, HeNormal
from tensorflow.keras.optimizers import Adam

import tensorflow.keras.backend as kb

dropout_rate = 0.05
l2_rate = 0.001


def euclidean_distance(vec):
	sum_2 = kb.sum(kb.square(vec[0]- vec[1]), axis=1, keepdims=True)
	return kb.sqrt(kb.maximum(sum_2, kb.epsilon()))

def contrastive_loss(y_true, y_pred):    
    # y_true = tf.cast(y_true, y_pred.dtype) # Should not be necessary

    # Calculate contrastive loss
    margin = 1
    # loss = y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
    loss = (1 - y_true) * tf.square(y_pred) +  y_true * tf.square(tf.maximum(margin - y_pred, 0))
    # loss = y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
    
    return tf.reduce_mean(loss)

def instanciate_siamese_model_(hp):
      def instanciate_embedding():
            input_layer = tf.keras.Input(INPUT_SHAPE)
            drop_conv = hp.Choice("drop_conv", values=[0.1])
            drop_dense = hp.Choice("drop_dense", values=[0.1])

            num_conv = hp.Choice("num_conv", values=[1, 2, 3])

            for i in range(1, num_conv + 1):
                  kernel_choice = hp.Choice(f"kernel_size{i}", values=[2])
                  x = Conv2D(filters=hp.Choice(f'filters{i}', values=[8, 16]),
                        kernel_size=(kernel_choice, kernel_choice),
                        kernel_initializer=HeNormal(seed=SEED),
                        activation='relu', padding="same")(input_layer)
                  x = tf.keras.layers.BatchNormalization()(x)
                  x = Dropout(drop_conv)(x) # 0.035
                  kernel_choice = hp.Choice(f"kernel_size{i+1}", values=[2])
                  x = Conv2D(filters=hp.Choice(f'filters{i+1}', values=[8, 16]),
                        kernel_size=(kernel_choice, kernel_choice),
                        kernel_initializer=HeNormal(seed=SEED),
                        activation='relu', padding="same")(x)
                  x = tf.keras.layers.BatchNormalization()(x)
                  pool_num_choice = hp.Choice(f"pool_kernel{i}", values=[2])
                  x = tf.keras.layers.MaxPool2D((pool_num_choice, pool_num_choice))(x)
                  x = Dropout(drop_conv)(x)
            
            x = tf.keras.layers.Flatten()(x)

            num_dense = hp.Choice("num_dense", values=[1, 2, 3, 4])

            for j in range(1, num_dense+1):
                  x = tf.keras.layers.Dense(
                        units=hp.Choice(f'units{j}', values=[32, 64, 128]),
                        activation="relu", kernel_initializer=HeNormal(seed=SEED)
                        )(x)
                  x = Dropout(drop_dense)(x)

            embedding_layer = tf.keras.models.Model(inputs=input_layer, outputs=x) # name=??
            return embedding_layer

            # return model
      
      embedding_layer = instanciate_embedding()

      input_1 = tf.keras.layers.Input(shape=INPUT_SHAPE)
      input_2 = tf.keras.layers.Input(shape=INPUT_SHAPE)

      embed_1 = embedding_layer(input_1)
      embed_2 = embedding_layer(input_2)

      # distance = tf.keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([embed_1, embed_2])
      distance = tf.keras.layers.Lambda(euclidean_distance, name="distance_layer")([embed_1, embed_2])

      # output = tf.keras.layers.Dense(1, activation='sigmoid', name="output_layer")(distance)

      model = tf.keras.models.Model(inputs=[input_1, input_2], outputs=distance)

      model.compile(
            # optimizer=Adam(learning_rate=hp.Choice("learning_rate", values=[0.01, 0.0001])),
            optimizer=hp.Choice("optimizer", values=["rmsprop", "adam"]),
            # optimizer = tf.keras.optimizers.RMSprop(learning_rate=SIA_LEARN_RATE),
            # optimizer='adam',
            loss=contrastive_loss, # NO-TE: Could try contrastive loss
            # loss="binary_crossentropy",
            metrics=["accuracy"])

      return model

# BEST WITH (128, 128) sized images (.61 accuracy)
def instanciate_siamese_model(hp):
      def instanciate_embedding():
            # model = tf.keras.models.Sequential()
            input_layer = tf.keras.Input(INPUT_SHAPE)
            # model.add(input_layer)
            drop_conv = hp.Choice("drop_conv", values=[0.1])
            drop_dense = hp.Choice("drop_dense", values=[0.1])
            # num_conv_layers = hp.Choice('num_conv_layers', values=[1, 2, 3])
            # num_dense_layers = hp.Choice('num_dense_layers', values=[0, 1, 2, 3])
            # kernel_choice = hp.Choice("kernel_size1", values=[2, 3, 5, 7])
            kernel_choice = hp.Choice("kernel_size1", values=[2])
            # x = Conv2D(filters=hp.Int('filters1', min_value=8, max_value=128, step=8),
            x = Conv2D(filters=hp.Choice('filters1', values=[8, 16]),
                  kernel_size=(kernel_choice, kernel_choice),
                  kernel_initializer=HeNormal(seed=SEED),
                  activation='relu', padding="same")(input_layer)
            x = tf.keras.layers.BatchNormalization()(x)
            x = Dropout(drop_conv)(x) # 0.035
            # kernel_choice = hp.Choice("kernel_size2", values=[2, 3, 5, 7])
            kernel_choice = hp.Choice("kernel_size2", values=[2])
            # x = Conv2D(filters=hp.Int('filters2', min_value=8, max_value=128, step=8),
            x = Conv2D(filters=hp.Choice('filters2', values=[8, 16]),
                  kernel_size=(kernel_choice, kernel_choice),
                  kernel_initializer=HeNormal(seed=SEED),
                  activation='relu', padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            # pool_num_choice = hp.Choice("pool_kernel1", values=[2, 3, 5, 7])
            pool_num_choice = hp.Choice("pool_kernel1", values=[2])
            x = tf.keras.layers.MaxPool2D((pool_num_choice, pool_num_choice))(x)
            x = Dropout(drop_conv)(x)
            # kernel_choice = hp.Choice("kernel_size1", values=[2, 3, 5, 7])
            # x = Conv2D(filters=hp.Int('filters1', min_value=8, max_value=128, step=8),
            # for i in range(1, num_conv_layers):
            #       if i == 1:
            #             model.add(Conv2D(filters=hp.Int(f'filters{i}', min_value=8, max_value=48, step=8),
            #                   # kernel_size=(kernel_choice, kernel_choice),
            #                   kernel_size=(2, 2),
            #                   kernel_initializer=HeNormal(seed=SEED),
            #                   activation='relu', padding="same"))
            #       else:
            #             model.add(Conv2D(filters=hp.Int(f'filters{i}', min_value=8, max_value=48, step=8),
            #                   # kernel_size=(kernel_choice, kernel_choice),
            #                   kernel_size=(2, 2),
            #                   kernel_initializer=HeNormal(seed=SEED),
            #                   activation='relu', padding="same"))
            #       model.add(tf.keras.layers.BatchNormalization())
            #       model.add(Dropout(drop_conv)) # 0.035
            #       # kernel_choice = hp.Choice("kernel_size2", values=[2, 3, 5, 7])
            #       # x = Conv2D(filters=hp.Int('filters2', min_value=8, max_value=128, step=8),
            #       model.add(Conv2D(filters=hp.Int(f'filters{i+1}', min_value=8, max_value=48, step=8),
            #             # kernel_size=(kernel_choice, kernel_choice),
            #             kernel_size=(2, 2),
            #             kernel_initializer=HeNormal(seed=SEED),
            #             activation='relu', padding="same"))
            #       model.add(tf.keras.layers.BatchNormalization())
            #       # pool_num_choice = hp.Choice("pool_kernel1", values=[2, 3, 5, 7])
            #       # x = tf.keras.layers.MaxPool2D((pool_num_choice, pool_num_choice))(x)
            #       model.add(tf.keras.layers.MaxPool2D((2, 2)))
            #       model.add(Dropout(drop_conv)) # 0.035

            # kernel_choice = hp.Choice("kernel_size3", values=[2, 3, 5, 7])
            kernel_choice = hp.Choice("kernel_size3", values=[2])
            # x = Conv2D(filters=hp.Int('filters3', min_value=8, max_value=128, step=8),
            x = Conv2D(filters=hp.Choice('filters3', values=[8, 16]),
                  kernel_size=(kernel_choice, kernel_choice),
                  kernel_initializer=HeNormal(seed=SEED),
                  activation='relu', padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = Dropout(drop_conv)(x) # 0.035
            # kernel_choice = hp.Choice("kernel_size4", values=[2, 3, 5, 7])
            kernel_choice = hp.Choice("kernel_size4", values=[2])
            # x = Conv2D(filters=hp.Int('filters4', min_value=8, max_value=128, step=8),
            x = Conv2D(filters=hp.Choice('filters4', values=[8, 16]),
                  kernel_size=(kernel_choice, kernel_choice),
                  kernel_initializer=HeNormal(seed=SEED),
                  activation='relu', padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            # pool_num_choice = hp.Choice("pool_kernel2", values=[2, 3, 5, 7])
            pool_num_choice = hp.Choice("pool_kernel2", values=[2])
            x = tf.keras.layers.MaxPool2D((pool_num_choice, pool_num_choice))(x)
            x = Dropout(drop_conv)(x)

            # kernel_choice = hp.Choice("kernel_size5", values=[2, 3, 5, 7])
            kernel_choice = hp.Choice("kernel_size5", values=[2])
            # x = Conv2D(filters=hp.Int('filters5', min_value=8, max_value=128, step=8),
            x = Conv2D(filters=hp.Choice('filters5', values=[8, 16]),
                  kernel_size=(kernel_choice, kernel_choice),
                  kernel_initializer=HeNormal(seed=SEED),
                  activation='relu', padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = Dropout(drop_conv)(x) # 0.035
            # kernel_choice = hp.Choice("kernel_size6", values=[2, 3, 5, 7])
            kernel_choice = hp.Choice("kernel_size6", values=[2])
            # x = Conv2D(filters=hp.Int('filters6', min_value=8, max_value=128, step=8),
            x = Conv2D(filters=hp.Choice('filters6', values=[8, 16]),
                  kernel_size=(kernel_choice, kernel_choice),
                  kernel_initializer=HeNormal(seed=SEED),
                  activation='relu', padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            # pool_num_choice = hp.Choice("pool_kernel3", values=[2, 3, 5, 7])
            pool_num_choice = hp.Choice("pool_kernel3", values=[2])
            x = tf.keras.layers.MaxPool2D((pool_num_choice, pool_num_choice))(x)
            x = Dropout(drop_conv)(x)
            
            x = tf.keras.layers.Flatten()(x)

            # model.add(tf.keras.layers.Flatten())
            # for j in range(1, num_dense_layers):
            #       model.add(tf.keras.layers.Dense(
            #             units=hp.Choice(f'units{j}', values=[32, 64, 96, 128]),
            #             activation="relu", kernel_initializer=HeNormal(seed=SEED)
            #             ))
            #       model.add(Dropout(drop_dense))
            x = tf.keras.layers.Dense(
                  # units=hp.Int('units2', min_value=32, max_value=256, step=32),
                  units=hp.Choice('units2', values=[32, 64]),
                  activation="relu", kernel_initializer=HeNormal(seed=SEED)
                  )(x)
            x = Dropout(drop_dense)(x)
            x = tf.keras.layers.Dense(
                  # units=hp.Int('units3', min_value=32, max_value=256, step=32),
                  units=hp.Choice('units3', values=[32, 64]),
                  activation="relu", kernel_initializer=HeNormal(seed=SEED)
                  )(x)
            x = Dropout(drop_dense)(x)
            x = tf.keras.layers.Dense(
                  # units=hp.Int('units4', min_value=32, max_value=256, step=32),
                  units=hp.Choice('units4', values=[32, 64]),
                  activation="relu", kernel_initializer=HeNormal(seed=SEED)
                  )(x)
            x = Dropout(drop_dense)(x)
            x = tf.keras.layers.Dense(
                  # units=hp.Int('units5', min_value=32, max_value=256, step=32),
                  units=hp.Choice('units5', values=[32, 64]),
                  activation="relu", kernel_initializer=HeNormal(seed=SEED)
                  )(x)
            # model.add(tf.keras.layers.Dense(
            #       units=2,
            #       activation=hp.Choice("activation_last", values=["relu", "softmax"]), kernel_initializer=HeNormal(seed=SEED)
            #       ))

            # x = tf.keras.layers.Dense(
            #       units=2,
            #       activation=hp.Choice("activation_last", values=["relu", "softmax"]), kernel_initializer=HeNormal(seed=SEED)
            #       )(x)

            x = tf.keras.layers.Dense(
                  units=2,
                  activation="relu", kernel_initializer=HeNormal(seed=SEED)
                  )(x)

            embedding_layer = tf.keras.models.Model(inputs=input_layer, outputs=x) # name=??
            return embedding_layer

            # return model
      
      embedding_layer = instanciate_embedding()

      input_1 = tf.keras.layers.Input(shape=INPUT_SHAPE)
      input_2 = tf.keras.layers.Input(shape=INPUT_SHAPE)

      embed_1 = embedding_layer(input_1)
      embed_2 = embedding_layer(input_2)

      # distance = tf.keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([embed_1, embed_2])
      distance = tf.keras.layers.Lambda(euclidean_distance, name="distance_layer")([embed_1, embed_2])

      # output = tf.keras.layers.Dense(1, activation='sigmoid', name="output_layer")(distance)

      model = tf.keras.models.Model(inputs=[input_1, input_2], outputs=distance)

      model.compile(
            # optimizer=Adam(learning_rate=SIA_LEARN_RATE),
            optimizer="adam",
            # optimizer = tf.keras.optimizers.RMSprop(learning_rate=SIA_LEARN_RATE),
            # optimizer='adam',
            loss=contrastive_loss, # NO-TE: Could try contrastive loss
            # loss="binary_crossentropy",
            metrics=["accuracy"])

      return model


def instanciate_sia_pca(hp):
      def instanciate_embedding():
            # model = tf.keras.models.Sequential()
            
            num_hidden = hp.Choice("num_hidden_layers", values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

            input_layer = tf.keras.Input(shape=(179, ))
            # model.add(input_layer)
            x = tf.keras.layers.Dense(hp.Choice("units_input", values=[32, 64, 96, 128]),
                  kernel_initializer=HeNormal(seed=SEED),
                  activation="relu")(input_layer)
            
            for i in range(1, num_hidden + 1):
                  x = tf.keras.layers.Dense(hp.Choice(f"units{i}", values=[32, 64, 96, 128]),
                  kernel_initializer=HeNormal(seed=SEED),
                  activation="relu")(x)

            x = tf.keras.layers.Dense(
                  units=2,
                  activation="relu", kernel_initializer=HeNormal(seed=SEED)
                  )(x)

            embedding_layer = tf.keras.models.Model(inputs=input_layer, outputs=x) # name=??
            return embedding_layer
            # return model
      
      embedding_layer = instanciate_embedding()

      input_1 = tf.keras.layers.Input(shape=(179, ))
      input_2 = tf.keras.layers.Input(shape=(179, ))

      embed_1 = embedding_layer(input_1)
      embed_2 = embedding_layer(input_2)

      # distance = tf.keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([embed_1, embed_2])
      distance = tf.keras.layers.Lambda(euclidean_distance, name="distance_layer")([embed_1, embed_2])

      # output = tf.keras.layers.Dense(1, activation='sigmoid', name="output_layer")(distance)

      model = tf.keras.models.Model(inputs=[input_1, input_2], outputs=distance)

      model.compile(
            # optimizer=Adam(learning_rate=SIA_LEARN_RATE),
            optimizer="adam",
            # optimizer = tf.keras.optimizers.RMSprop(learning_rate=SIA_LEARN_RATE),
            # optimizer='adam',
            loss=contrastive_loss, # NO-TE: Could try contrastive loss
            # loss="binary_crossentropy",
            metrics=["accuracy"])

      return model

# sia_pca = instanciate_sia_pca()


# def instanciate_siamese_model():
#       input_layer = tf.keras.Input(INPUT_SHAPE)
#       # x = Conv2D(filters=64, kernel_size=(2, 2), 
#       #         #    kernel_initializer=GlorotNormal(seed=None),
#       #         #    kernel_regularizer=regularizers.l2(0.1), 
#       #            activation='relu', padding="same", input_shape=(300, 300, 1))(input_layer)
#       # x = tf.keras.layers.BatchNormalization()(x)
#       # x = Conv2D(filters=64, kernel_size=(2, 2), 
#       #         #    kernel_initializer=GlorotNormal(seed=None),
#       #         #    kernel_regularizer=regularizers.l2(0.1), 
#       #            activation='relu', padding="same")(x)
#       # x = tf.keras.layers.BatchNormalization()(x)
#       # x = tf.keras.layers.MaxPool2D((2, 2))(x)
#       # x = Dropout(0.1)(x) # 0.035
#       #    x = Conv2D(filters=64, kernel_size=(2, 2), 
#       #             #   kernel_initializer=GlorotNormal(seed=None),
#       #             kernel_initializer=HeNormal(seed=SEED),
#       #          #    kernel_regularizer=regularizers.l2(0.1), 
#       #             activation='relu', padding="same", input_shape=INPUT_SHAPE)(input_layer)
#       #    x = tf.keras.layers.BatchNormalization()(x)
#       #    x = Conv2D(filters=32, kernel_size=(2, 2), 
#       #             #   kernel_initializer=GlorotNormal(seed=None),
#       #             kernel_initializer=HeNormal(seed=SEED),
#       #          #    kernel_regularizer=regularizers.l2(0.1), 
#       #             activation='relu', padding="same")(x)
#       #    x = tf.keras.layers.BatchNormalization()(x)
#       #    x = tf.keras.layers.MaxPool2D((2, 2))(x)
#       #    # x = Dropout(0.1)(x) # 0.035
#       #    # x = Conv2D(filters=16, kernel_size=(2, 2), 
#       #    #          #   kernel_initializer=GlorotNormal(seed=None),
#       #    #          kernel_initializer=HeNormal(seed=SEED),
#       #    #       #    kernel_regularizer=regularizers.l2(0.1), 
#       #    #          activation='relu', padding="same")(x)
#       #    # x = tf.keras.layers.BatchNormalization()(x)
#       #    # x = Conv2D(filters=16, kernel_size=(2, 2), 
#       #    #          #   kernel_initializer=GlorotNormal(seed=None),
#       #    #          kernel_initializer=HeNormal(seed=SEED),
#       #    #       #    kernel_regularizer=regularizers.l2(0.1), 
#       #    #          activation='relu', padding="same")(x)
#       #    # x = tf.keras.layers.BatchNormalization()(x)
#       #    # x = tf.keras.layers.MaxPool2D((2, 2))(x)
#       #    # # x = Dropout(0.1)(x) # 0.035
#       #    # x = Conv2D(filters=8, kernel_size=(2, 2), 
#       #    #          #   kernel_initializer=GlorotNormal(seed=None),
#       #    #          kernel_initializer=HeNormal(seed=SEED),
#       #    #       #    kernel_regularizer=regularizers.l2(0.1), 
#       #    #          activation='relu', padding="same")(x)
#       #    # x = tf.keras.layers.BatchNormalization()(x)
#       #    # x = Conv2D(filters=8, kernel_size=(2, 2), 
#       #    #          #   kernel_initializer=GlorotNormal(seed=None),
#       #    #          kernel_initializer=HeNormal(seed=SEED),
#       #    #       #    kernel_regularizer=regularizers.l2(0.1), 
#       #    #          activation='relu', padding="same")(x)
#       #    # x = tf.keras.layers.BatchNormalization()(x)
#       #    # x = tf.keras.layers.MaxPool2D((2, 2))(x)
#       #    # x = Dropout(0.1)(x) # 0.035
#       #    x = tf.keras.layers.Flatten()(x)
#       #    x = tf.keras.layers.Dense(1024,
#       #                            #   kernel_initializer=GlorotNormal(seed=None),
#       #                      kernel_initializer=HeNormal(seed=SEED),
#       #                            #   kernel_regularizer=regularizers.l2(l2_rate), 
#       #                            activation="relu")(x)
#       #    x = tf.keras.layers.Dense(512,
#       #                            #   kernel_initializer=GlorotNormal(seed=None),
#       #                      kernel_initializer=HeNormal(seed=SEED),
#       #                            #   kernel_regularizer=regularizers.l2(l2_rate), 
#       #                            activation="relu")(x)
#       #    x = tf.keras.layers.Dense(512,
#       #                            #   kernel_initializer=GlorotNormal(seed=None),
#       #                      kernel_initializer=HeNormal(seed=SEED),
#       #                            #   kernel_regularizer=regularizers.l2(l2_rate), 
#       #                            activation="relu")(x)
#       #    embedding_layer = tf.keras.models.Model(inputs=input_layer, outputs=x)
#       #    embedding_layer.summary()

#       x = Conv2D(filters=32, kernel_size=(2, 2), 
#             #   kernel_initializer=GlorotNormal(seed=None),
#             kernel_initializer=HeNormal(seed=SEED),
#             #    kernel_regularizer=regularizers.l2(0.1), 
#             activation='relu', padding="same", input_shape=INPUT_SHAPE)(input_layer)
#       x = tf.keras.layers.BatchNormalization()(x)
#       # x = tf.keras.layers.MaxPool2D((2, 2))(x)
#       x = Dropout(0.1)(x) # 0.035
#       x = Conv2D(filters=16, kernel_size=(2, 2), 
#             #   kernel_initializer=GlorotNormal(seed=None),
#             kernel_initializer=HeNormal(seed=SEED),
#             #    kernel_regularizer=regularizers.l2(0.1), 
#             activation='relu', padding="same")(x)
#       x = tf.keras.layers.BatchNormalization()(x)
#       x = tf.keras.layers.MaxPool2D((2, 2))(x)
#       x = Dropout(0.1)(x) # 0.035
#       x = Conv2D(filters=8, kernel_size=(2, 2), 
#             #   kernel_initializer=GlorotNormal(seed=None),
#             kernel_initializer=HeNormal(seed=SEED),
#             #    kernel_regularizer=regularizers.l2(0.1), 
#             activation='relu', padding="same")(x)
#       x = tf.keras.layers.BatchNormalization()(x)
#       # x = tf.keras.layers.MaxPool2D((2, 2))(x)
#       x = Dropout(0.1)(x) # 0.035
#       x = Conv2D(filters=4, kernel_size=(2, 2), 
#             #   kernel_initializer=GlorotNormal(seed=None),
#             kernel_initializer=HeNormal(seed=SEED),
#             #    kernel_regularizer=regularizers.l2(0.1), 
#             activation='relu', padding="same")(x)
#       x = tf.keras.layers.BatchNormalization()(x)
#       x = tf.keras.layers.MaxPool2D((2, 2))(x)
#       x = Dropout(0.1)(x) # 0.035
#       x = tf.keras.layers.Flatten()(x)
#       x = tf.keras.layers.Dense(256,
#                               #   kernel_initializer=GlorotNormal(seed=None),
#                         kernel_initializer=HeNormal(seed=SEED),
#                               #   kernel_regularizer=regularizers.l2(l2_rate), 
#                               activation="relu")(x)
#       x = Dropout(0.2)(x) # 0.035
#       x = tf.keras.layers.Dense(128,
#                               #   kernel_initializer=GlorotNormal(seed=None),
#                         kernel_initializer=HeNormal(seed=SEED),
#                               #   kernel_regularizer=regularizers.l2(l2_rate), 
#                               activation="relu")(x)
#       embedding_layer = tf.keras.models.Model(inputs=input_layer, outputs=x) # name=??
#       embedding_layer.summary()

#       input_1 = tf.keras.layers.Input(shape=INPUT_SHAPE)
#       input_2 = tf.keras.layers.Input(shape=INPUT_SHAPE)

#       embed_1 = embedding_layer(input_1)
#       embed_2 = embedding_layer(input_2)

#       # distance = tf.keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([embed_1, embed_2])
#       distance = tf.keras.layers.Lambda(euclidean_distance, name="distance_layer")([embed_1, embed_2])

#       # output = tf.keras.layers.Dense(1, activation='sigmoid', name="output_layer")(distance)

#       model = tf.keras.models.Model(inputs=[input_1, input_2], outputs=distance)

#       return model

# model = instanciate_siamese_model()
# model.summary()

