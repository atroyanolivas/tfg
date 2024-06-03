import sys

sys.path.append("./")

from config import PCA_LEARN_RATE, PCA_EPOCHS, PCA_BATCH_SIZE, PCA_TARGET_SIZE, N_COMP, TRAIN_SPLIT, VAL_SPLIT

import numpy as np
import math
from pca.model import build_pca_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from keras_tuner.tuners import RandomSearch


X_train = np.load('./dataset/arrays/X_train_2_class.npy')
Y_train = np.load('./dataset/arrays/Y_train_2_class.npy')
Y_train = (Y_train[:, 1] == 1).astype(int) # 0: tumour, 1: notumour

X_val = np.load('./dataset/arrays/X_val_2_class.npy')
Y_val = np.load('./dataset/arrays/Y_val_2_class.npy')
Y_val = (Y_val[:, 1] == 1).astype(int)

X_test = np.load('./dataset/arrays/X_test_2_class.npy')
Y_test = np.load('./dataset/arrays/Y_test_2_class.npy')
Y_test = (Y_test[:, 1] == 1).astype(int)

print("X_train.shape =", X_train.shape)
print("Y_train.shape =", Y_train.shape)

print("X_val.shape =", X_val.shape)
print("Y_val.shape =", Y_val.shape)

print("X_test.shape =", X_test.shape)
print("Y_test.shape =", Y_test.shape)

all_dataset = np.concatenate((X_train, X_val, X_test), axis=0)

l_train = len(X_train)
l_val = len(X_val)
l_test = len(X_test)

# batch = (BATCH_SIZE, im_height, im_width, num_channels)
def apply_PCA_to_batch(batch, n_components=228): # 335
    original_shape = batch.shape
    first_channel = batch[:, :, :, 0] # Separate the channels, since there is only one
    print("first_channel.shape =", first_channel.shape)
    observation_matrix = np.reshape(first_channel, (-1, original_shape[1] * original_shape[2]))
    print("observation_matrix.shape =", observation_matrix.shape)

    scaler = StandardScaler()
    norm_obs = scaler.fit_transform(observation_matrix)

    pca = PCA(n_components)
    transformed_batch = pca.fit_transform(norm_obs)

    return transformed_batch



transformed_data = apply_PCA_to_batch(all_dataset)
print("transformed_data.shape =", transformed_data.shape)

X_train = transformed_data[:l_train, :]
X_val = transformed_data[l_train:l_train + l_val, :]
X_test = transformed_data[l_train + l_val:l_train + l_val + l_test, :]

# np.save("./test/arrays/X_train_pca.npy", X_train)
# np.save("./test/arrays/X_val_pca.npy", X_val)
# np.save("./test/arrays/X_test_pca.npy", X_test)

print("X_train.shape =", X_train.shape)
print("X_val.shape =", X_val.shape)
print("X_test.shape =", X_test.shape)

tuner = RandomSearch(
    build_pca_model,
    objective="val_accuracy",
    max_trials=50,
    directory="./pca/",
    project_name="pcamlp_tuning_random"
)

tuner.search(X_train, Y_train, epochs=65, # 65
            validation_data=(X_val, Y_val))

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)
# model_tf.fit(X_train, y_train, epochs=PCA_EPOCHS, batch_size=10, validation_data=(X_test, y_test))

print("Best hyperparameters:")
for hp in best_hps.values.keys():
    print(f"{hp}: {best_hps.get(hp)}")
best_model = tuner.hypermodel.build(best_hps)

best_model.summary()

# Fit the data (perform the training)
history = best_model.fit(
    x=X_train,
    y=Y_train,
    epochs=PCA_EPOCHS,
    batch_size=PCA_BATCH_SIZE,
    
    shuffle=True,

    validation_data=(X_val, Y_val),
    validation_batch_size=PCA_BATCH_SIZE,
    verbose=1
)

# Save the model in "./model" directory
best_model.save("./test/models/PCA_MLP.keras")

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