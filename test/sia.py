import tensorflow as tf
import numpy as np

import sys

sys.path.append("./")
from config import SEED

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from test.main import instanciate_siamese_conv

from keras_tuner.tuners import RandomSearch

sia_conv = instanciate_siamese_conv()

X_train = np.load('./dataset/arrays/X_train_2_class.npy')
Y_train = np.load('./dataset/arrays/Y_train_2_class.npy')
Y_train = (Y_train[:, 1] == 1).astype(int) # 0: tumour, 1: notumour

X_val = np.load('./dataset/arrays/X_val_2_class.npy')
Y_val = np.load('./dataset/arrays/Y_val_2_class.npy')
Y_val = (Y_val[:, 1] == 1).astype(int)

X_train_maped = sia_conv.predict(X_train)
X_val_maped = sia_conv.predict(X_val)

def create_mlp_model(hp):
    model = tf.keras.models.Sequential()
    model.add(Input((128,)))
    model.add(Dense(hp.Choice("units1", values=[32, 64, 128]), activation='relu'))
    model.add(Dense(hp.Choice("units2", values=[32, 64, 128]), activation='relu'))
    model.add(Dense(hp.Choice("units3", values=[32, 64, 128]), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # For binary classification

    model.compile(optimizer=Adam(learning_rate=hp.Choice("learning_rate", values=[0.01, 0.001])),
              loss='binary_crossentropy',
              metrics=['accuracy'])
    
    return model


tuner = RandomSearch(
    create_mlp_model,
    objective="val_accuracy",
    max_trials=30,
    directory="./test/",
    # project_name="conv_tuning_2_class_fine_tune"
    project_name="sia-mlp_tunning"
)

tuner.search(X_train_maped, Y_train, epochs=50, validation_data=(X_val_maped, Y_val))

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters:")
for hp in best_hps.values.keys():
    print(f"{hp}: {best_hps.get(hp)}")
best_model = tuner.hypermodel.build(best_hps)

# Fit the data (perform the training)
history = best_model.fit(
    x=X_train_maped,
    y=Y_train,
    epochs=65,
    batch_size=32,
    
    shuffle=True,

    validation_data=(X_val_maped, Y_val),
    validation_batch_size=32,
    verbose=1
)


# Save the model in "./model" directory
best_model.save("./test/models/sia-mlp.keras")

import matplotlib.pyplot as plt

plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  

epochs = range(len(history.history["accuracy"]))

## Plot accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(epochs, history.history["accuracy"], colors[0], label='Training accuracy')
ax1.plot(epochs, history.history["val_accuracy"], colors[1], label='Validation accuracy')
ax1.set_title('Training and validation accuracy')
ax1.grid()
ax1.legend(loc="lower right")

## Plot Loss
ax2.plot(epochs, history.history["loss"], colors[4], label='Training Loss')
ax2.plot(epochs, history.history["val_loss"], colors[5], label='Validation Loss')
ax2.set_title('Training and validation Loss')
ax2.grid()
ax2.legend(loc="upper right")

plt.show()
