import sys

sys.path.append("./")
from convolutional.model import build_conv_model
import tensorflow as tf
from config import BATCH_SIZE, SEED

from tensorflow.keras.layers import Conv2D, Dropout, Dense
import numpy as np
from tensorflow.keras.initializers import HeNormal

from keras_tuner.tuners import RandomSearch
from tensorflow.keras.optimizers import Adam


X_train = np.load('./kaggle/Data/X_train.npy')
Y_train = np.load('./kaggle/Data/Y_train.npy')
Y_train = (Y_train[:, 1] == 1).astype(int)

X_val = np.load('./kaggle/Data/X_val.npy')
Y_val = np.load('./kaggle/Data/Y_val.npy')
Y_val = (Y_val[:, 1] == 1).astype(int)


# def build_conv_model(hp):
#    model = tf.keras.models.Sequential()
#    pool_num_choice = hp.Choice("pool_kernel", values=[2, 3])
#    pool_choice = (pool_num_choice, pool_num_choice)
#    drop_conv = hp.Choice("drop_conv", values=[0.1, 0.2])
#    drop_dense = hp.Choice("drop_dense", values=[0.1, 0.2])

#    # 1st block
#    kernel_choice = hp.Choice("kernel_size1", values=[2, 3, 5, 7])
#    model.add(
#       Conv2D(filters=hp.Int('filters1', min_value=8, max_value=128, step=8),
#          kernel_size=(kernel_choice, kernel_choice),
#          kernel_initializer=HeNormal(seed=SEED),
#          activation='relu', padding="same"))
#    model.add(tf.keras.layers.BatchNormalization())
#    kernel_choice = hp.Choice("kernel_size2", values=[2, 3, 5, 7])
#    model.add(
#       Conv2D(filters=hp.Int('filters2', min_value=8, max_value=128, step=8),
#          kernel_size=(kernel_choice, kernel_choice),
#          kernel_initializer=HeNormal(seed=SEED),
#          activation='relu', padding="same"))
#    model.add(tf.keras.layers.BatchNormalization())
#    model.add(tf.keras.layers.MaxPool2D(pool_choice))
#    model.add(Dropout(drop_conv))
#    kernel_choice = hp.Choice("kernel_size3", values=[2, 3, 5, 7])
#    # kernel_choice = hp.Choice("kernel_size3", values=[2])
#    model.add(
#       Conv2D(filters=hp.Int('filters3', min_value=8, max_value=128, step=8),
#       # Conv2D(filters=hp.Choice("filters3", values=[2, 4, 6, 8, 12, 16]),
#          kernel_size=(kernel_choice, kernel_choice),
#          kernel_initializer=HeNormal(seed=SEED),
#         #    kernel_regularizer=regularizers.l2(0.1), 
#          activation='relu', padding="same"))
#    model.add(tf.keras.layers.BatchNormalization())
#    kernel_choice = hp.Choice("kernel_size4", values=[2, 3, 5, 7])
#    # kernel_choice = hp.Choice("kernel_size4", values=[2])
#    model.add(
#       Conv2D(filters=hp.Int('filters4', min_value=8, max_value=128, step=8),
#       # Conv2D(filters=hp.Choice("filters4", values=[2, 4, 6, 8, 12, 16]),
#          kernel_size=(kernel_choice, kernel_choice),
#          kernel_initializer=HeNormal(seed=SEED),
#         #    kernel_regularizer=regularizers.l2(0.1), 
#          activation='relu', padding="same"))
#    model.add(tf.keras.layers.BatchNormalization())
#    # pool_num_choice = hp.Choice("pool_kernel2", values=[2, 3])
#    model.add(tf.keras.layers.MaxPool2D((pool_num_choice, pool_num_choice)))
#    model.add(Dropout(drop_conv))
#    # 3rd block
#    kernel_choice = hp.Choice("kernel_size5", values=[2, 3, 5, 7])
#    # kernel_choice = hp.Choice("kernel_size5", values=[2])
#    model.add(
#       Conv2D(filters=hp.Int('filters5', min_value=8, max_value=128, step=8),
#       # Conv2D(filters=hp.Choice("filters5", values=[2, 4, 6, 8, 12, 16]),
#          kernel_size=(kernel_choice, kernel_choice),
#          kernel_initializer=HeNormal(seed=SEED),
#         #    kernel_regularizer=regularizers.l2(0.1), 
#          activation='relu', padding="same"))
#    model.add(tf.keras.layers.BatchNormalization())
#    kernel_choice = hp.Choice("kernel_size6", values=[2, 3, 5, 7])
#    # kernel_choice = hp.Choice("kernel_size6", values=[2])
#    model.add(
#       Conv2D(filters=hp.Int('filters6', min_value=8, max_value=128, step=8),
#       # Conv2D(filters=hp.Choice("filters6", values=[2, 4, 6, 8, 12, 16]),
#          kernel_size=(kernel_choice, kernel_choice),
#          kernel_initializer=HeNormal(seed=SEED),
#         #    kernel_regularizer=regularizers.l2(0.1), 
#          activation='relu', padding="same"))
#    model.add(tf.keras.layers.BatchNormalization())
#    model.add(tf.keras.layers.MaxPool2D(pool_choice))
#    model.add(Dropout(drop_conv))
   


#    model.add(tf.keras.layers.Flatten())
#    model.add(Dense(
#         units=hp.Int('units1', min_value=32, max_value=256, step=32),
#          # units=hp.Choice('units1', values=[8, 16, 32, 48, 64]),
#          activation="relu", kernel_initializer=HeNormal(seed=SEED)
#    ))
#    model.add(Dropout(drop_dense))
#    model.add(Dense(
#         units=hp.Int('units2', min_value=32, max_value=256, step=32),
#          # units=hp.Choice('units2', values=[8, 16, 32, 64, 72, 96, 104]),
#          activation="relu", kernel_initializer=HeNormal(seed=SEED)
#    ))
#    model.add(Dropout(drop_dense))
#    model.add(Dense(
#         units=hp.Int('units3', min_value=32, max_value=256, step=32),
#          # units=hp.Choice('units3', values=[8, 16, 96, 104, 112, 120, 128]),
#          activation="relu", kernel_initializer=HeNormal(seed=SEED)
#    ))
#    model.add(Dropout(drop_dense))
#    model.add(Dense(1, activation="sigmoid"))

#    model.compile(
#       optimizer=Adam(learning_rate=hp.Choice("learning_rate", values=[0.01, 0.001, 0.0001, 0.00001])),
#       # optimizer=Adam(learning_rate=hp.Choice("learning_rate", values=[0.01, 0.001])),
#       loss="binary_crossentropy",
#       metrics=["accuracy"]
#    )

#    return model


tuner = RandomSearch(
    build_conv_model,
    objective="val_accuracy",
    max_trials=100,
    directory="./kaggle/",
    project_name="conv_random1"
)

# tuner.search(X_train, Y_train, epochs=30, validation_data=(X_val, Y_val))

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

print("Best hyperparameters:")
for hp in best_hps.values.keys():
    print(f"{hp}: {best_hps.get(hp)}")
best_model = tuner.hypermodel.build(best_hps)

# Fit the data (perform the training)
history = best_model.fit(
    x=X_train,
    y=Y_train,
    epochs=12,
    batch_size=BATCH_SIZE,
    
    shuffle=True,

    validation_data=(X_val, Y_val),
    validation_batch_size=BATCH_SIZE,
    verbose=1
)

best_model.save("./kaggle/models/conv.keras")

