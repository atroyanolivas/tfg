import sys

sys.path.append("./")

from config import LEARN_RATE, CONV_EPOCHS, BATCH_SIZE, IMG_SIZE

from convolutional.model import build_conv_model
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np

from keras_tuner.tuners import RandomSearch

X_train = np.load('./dataset/arrays/X_train_2_class.npy')
Y_train = np.load('./dataset/arrays/Y_train_2_class.npy')
Y_train = (Y_train[:, 1] == 1).astype(int)

X_val = np.load('./dataset/arrays/X_val_2_class.npy')
Y_val = np.load('./dataset/arrays/Y_val_2_class.npy')
Y_val = (Y_val[:, 1] == 1).astype(int)

tuner = RandomSearch(
    build_conv_model,
    objective="val_accuracy",
    max_trials=60,
    directory="./convolutional/",
    project_name="conv_tuning_2_class_random_extra"
)

# tuner.search(X_train, Y_train, epochs=64, validation_data=(X_val, Y_val))

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters:")
for hp in best_hps.values.keys():
    print(f"{hp}: {best_hps.get(hp)}")
best_model = tuner.hypermodel.build(best_hps)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=8,
    mode='max',
    baseline=0.83,
    restore_best_weights=True
)

# Fit the data (perform the training)
history = best_model.fit(
    x=X_train,
    y=Y_train,
    epochs=16,
    batch_size=BATCH_SIZE,
    
    shuffle=True,

    validation_data=(X_val, Y_val),
    validation_batch_size=BATCH_SIZE,
    verbose=1
    # callbacks=[early_stopping]
)

# _, accuracy = best_model.evaluate(best_model.predict(X_val), Y_val)
# print("Validation accuracy =", accuracy)

# Save the model in "./model" directory
best_model.save("./test/models/conv_2_class_3.keras")

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

# plt.show()