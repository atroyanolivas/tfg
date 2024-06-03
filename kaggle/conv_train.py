import sys

sys.path.append("./")

from config import LEARN_RATE, CONV_EPOCHS, BATCH_SIZE, SEED

import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Dropout, Dense

from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam

import numpy as np

# from tensorflow.keras.optimizers import RMSprop

from keras_tuner.tuners import GridSearch


X_train = np.load('./kaggle/Data/X_train.npy')
print("X_train.shape ", X_train.shape)
Y_train = np.load('./kaggle/Data/Y_train.npy')
print("Y_train.shape ", Y_train.shape)

X_val = np.load('./kaggle/Data/X_val.npy')
print("X_val.shape ", X_val.shape)
Y_val = np.load('./kaggle/Data/Y_val.npy')
print("Y_val.shape ", Y_val.shape)



def build_conv_model(hp):
   model = tf.keras.models.Sequential()
   pool_num_choice = hp.Choice("pool_kernel", values=[2, 3, 5, 7])
   pool_choice = (pool_num_choice, pool_num_choice)
   drop_conv = hp.Choice("drop_conv", values=[0.1, 0.2])
   drop_dense = hp.Choice("drop_dense", values=[0.1, 0.2])
   # 1st block
   kernel_choice = hp.Choice("kernel_size1", values=[2, 3, 5, 7])
   model.add(
      Conv2D(filters=hp.Int('filters1', min_value=8, max_value=128, step=8),
         kernel_size=(kernel_choice, kernel_choice),
         kernel_initializer=HeNormal(seed=SEED),
        #    kernel_regularizer=regularizers.l2(0.1), 
         activation='relu', padding="same"))
   model.add(tf.keras.layers.BatchNormalization())
   kernel_choice = hp.Choice("kernel_size2", values=[2, 3, 5, 7])
   model.add(
      Conv2D(filters=hp.Int('filters2', min_value=8, max_value=128, step=8),
         kernel_size=(kernel_choice, kernel_choice),
         kernel_initializer=HeNormal(seed=SEED),
        #    kernel_regularizer=regularizers.l2(0.1), 
         activation='relu', padding="same"))
   model.add(tf.keras.layers.BatchNormalization())
   model.add(tf.keras.layers.MaxPool2D(pool_choice))
   model.add(Dropout(drop_conv))
   # 2nd block
   kernel_choice = hp.Choice("kernel_size3", values=[2, 3, 5, 7])
   model.add(
      Conv2D(filters=hp.Int('filters3', min_value=8, max_value=128, step=8),
         kernel_size=(kernel_choice, kernel_choice),
         kernel_initializer=HeNormal(seed=SEED),
        #    kernel_regularizer=regularizers.l2(0.1), 
         activation='relu', padding="same"))
   model.add(tf.keras.layers.BatchNormalization())
   kernel_choice = hp.Choice("kernel_size4", values=[2, 3, 5, 7])
   model.add(
      Conv2D(filters=hp.Int('filters4', min_value=8, max_value=128, step=8),
         kernel_size=(kernel_choice, kernel_choice),
         kernel_initializer=HeNormal(seed=SEED),
        #    kernel_regularizer=regularizers.l2(0.1), 
         activation='relu', padding="same"))
   model.add(tf.keras.layers.BatchNormalization())
   model.add(tf.keras.layers.MaxPool2D(pool_choice))
   model.add(Dropout(drop_conv))
   # 3rd block
   kernel_choice = hp.Choice("kernel_size5", values=[2, 3, 5, 7])
   model.add(
      Conv2D(filters=hp.Int('filters5', min_value=8, max_value=128, step=8),
         kernel_size=(kernel_choice, kernel_choice),
         kernel_initializer=HeNormal(seed=SEED),
        #    kernel_regularizer=regularizers.l2(0.1), 
         activation='relu', padding="same"))
   model.add(tf.keras.layers.BatchNormalization())
   kernel_choice = hp.Choice("kernel_size6", values=[2, 3, 5, 7])
   model.add(
      Conv2D(filters=hp.Int('filters6', min_value=8, max_value=128, step=8),
         kernel_size=(kernel_choice, kernel_choice),
         kernel_initializer=HeNormal(seed=SEED),
        #    kernel_regularizer=regularizers.l2(0.1), 
         activation='relu', padding="same"))
   model.add(tf.keras.layers.BatchNormalization())
   model.add(tf.keras.layers.MaxPool2D(pool_choice))
   model.add(Dropout(drop_conv))
   model.add(tf.keras.layers.Flatten())
   model.add(Dense(
        units=hp.Int('units1', min_value=32, max_value=256, step=32),
        activation="relu", kernel_initializer=HeNormal(seed=SEED)
   ))
   model.add(Dropout(drop_dense))
   model.add(Dense(
        units=hp.Int('units2', min_value=32, max_value=256, step=32),
        activation="relu", kernel_initializer=HeNormal(seed=SEED)
   ))
   model.add(Dropout(drop_dense))
   model.add(Dense(
        units=hp.Int('units3', min_value=32, max_value=256, step=32),
        activation="relu", kernel_initializer=HeNormal(seed=SEED)
   ))
   model.add(Dropout(drop_dense))
   model.add(Dense(2, activation="softmax"))

   model.compile(
      optimizer=Adam(learning_rate=hp.Choice("learning_rate", values=[0.01, 0.001, 0.0001, 0.00001])),
      loss="categorical_crossentropy",
      metrics=["accuracy"]
   )

   return model




tuner = GridSearch(
    build_conv_model,
    objective="val_accuracy",
    max_trials=20,
    directory="./kaggle/",
    project_name="conv_tuning"
)

tuner.search(X_train, Y_train, epochs=64, validation_data=(X_val, Y_val))

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

# Fit the data (perform the training)
history = best_model.fit(
    x=X_train,
    y=Y_train,
    epochs=90,
    batch_size=BATCH_SIZE,
    
    shuffle=True,

    validation_data=(X_val, Y_val),
    validation_batch_size=BATCH_SIZE,
    verbose=1
)
# Save the model in "./model" directory
best_model.save("./model/vgg8.keras")

import matplotlib.pyplot as plt

# plt.style.use("tableau-colorblind10")
plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  

epochs = range(len(history.history["accuracy"]))

## Plot accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(epochs, history.history["accuracy"], colors[0], label='Training accuracy')
ax1.plot(epochs, history.history["val_accuracy"], colors[1], label='Validation accuracy')
# ax2.plot([0, 0], [epochs, epochs], color="white")
# ax2.plot([epochs, epochs], [0, 0], color="white")
ax1.set_title('Training and validation accuracy')
ax1.grid()
ax1.legend(loc="lower right")

## Plot Loss
ax2.plot(epochs, history.history["loss"], colors[4], label='Training Loss')
ax2.plot(epochs, history.history["val_loss"], colors[5], label='Validation Loss')
# ax2.plot([0, 0], [0, epochs], color="white")
# ax2.plot([0, epochs], [0, 0], color="white")
ax2.set_title('Training and validation Loss')
ax2.grid()
ax2.legend(loc="upper right")

plt.show()