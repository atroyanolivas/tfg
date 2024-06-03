import sys

sys.path.append("./")
from config import SIA_EPOCHS, SIA_BATCH_SIZE, SIA_LEARN_RATE, IMG_SIZE, TRAIN_SPLIT, VAL_SPLIT
from siamese.model import instanciate_sia_pca

import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

from keras_tuner.tuners import GridSearch



X1 = np.load('./dataset/arrays/siamese/X1_train_2_class.npy')
len_x1 = len(X1)
print("X1.shape =", X1.shape)
X2 = np.load('./dataset/arrays/siamese/X2_train_2_class.npy')
len_x2 = len(X2)
print("X2.shape =", X2.shape)
Y = np.load('./dataset/arrays/siamese/Y_train_2_class.npy')

X_val1 = np.load('./dataset/arrays/siamese/X_val1_2_class.npy')
len_xv1 = len(X_val1)
print("X_val1.shape =", X_val1.shape)
X_val2 = np.load('./dataset/arrays/siamese/X_val2_2_class.npy')
len_xv2 = len(X_val2)
print("X_val2.shape =", X_val2.shape)
Y_val = np.load('./dataset/arrays/siamese/Y_val_2_class.npy')


all_dataset = np.concatenate((X1, X2, X_val1, X_val2), axis=0)
# all_dataset = (all_dataset - np.mean(all_dataset, axis=0)) / np.std(all_dataset, axis=0)


def apply_PCA_to_batch(batch, n_components=179):
    original_shape = batch.shape
    first_channel = batch[:, :, :, 0] # Separate the channels, since there is only one
    print("first_channel.shape =", first_channel.shape)
    observation_matrix = np.reshape(first_channel, (-1, original_shape[1] * original_shape[2]))
    print("observation_matrix.shape =", observation_matrix.shape)

    scaler = StandardScaler()
    norm_obs = scaler.fit_transform(observation_matrix)

    pca = PCA(n_components)
    transformed_batch = pca.fit_transform(norm_obs) # Shape should be: (BATCH_SIZE, im_height*im_width)

    return transformed_batch
    # return observation_matrix

transformed_data = apply_PCA_to_batch(all_dataset)
print("transformed_data.shape =", transformed_data.shape)

X1 = transformed_data[:len_x1, :]
X2 = transformed_data[len_x1:len_x1 + len_x2, :]

X_val1 = transformed_data[len_x1 + len_x2: len_x1 + len_x2 + len_xv1, :]
X_val2 = transformed_data[len_x1 + len_x2 + len_xv1: len_x1 + len_x2 + len_xv1 + len_xv2, :]

def contrastive_loss(y_true, y_pred):    
    # y_true = tf.cast(y_true, y_pred.dtype) # Should not be necessary

    # Calculate contrastive loss
    margin = 1
    # loss = y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
    loss = (1 - y_true) * tf.square(y_pred) +  y_true * tf.square(tf.maximum(margin - y_pred, 0))
    # loss = y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
    
    return tf.reduce_mean(loss)

tuner = GridSearch(
    instanciate_sia_pca,
    objective="val_accuracy",
    max_trials=100,
    directory="./siamese/",
    project_name="sia_pca_2_class"
)

tuner.search([X1, X2], Y, epochs=65, validation_data=([X_val1, X_val2], Y_val))

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

history = best_model.fit(
    x=[X1, X2],
    y=Y,
    epochs=80,
    batch_size=SIA_BATCH_SIZE,
    
    shuffle=True,

    validation_data=([X_val1, X_val2], Y_val),
    validation_batch_size=SIA_BATCH_SIZE,
    verbose=1
)

# Save the model
best_model.save("./model/siamese_best.keras")
best_model.save_weights("./model/siamese_best_weights.weights.h5")

# Save the model
best_model.save("./model/siamese_pca.keras")
best_model.save_weights("./model/siamese_pca_weights.weights.h5")

import matplotlib.pyplot as plt

# plt.style.use("tableau-colorblind10")
plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  

epochs = range(len(history.history["accuracy"]))

## Plot accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(epochs, history.history["accuracy"], colors[0], label='Training accuracy')
ax1.plot(epochs, history.history["val_accuracy"], colors[1], label='Validation accuracy')
ax1.grid()
ax1.set_title('Training and validation accuracy')
ax1.legend(loc="upper left")

## Plot Loss
ax2.plot(epochs, history.history["loss"], colors[4], label='Training Loss')
ax2.plot(epochs, history.history["val_loss"], colors[5], label='Validation Loss')
ax2.grid()
ax2.set_title('Training and validation Loss')
ax2.legend(loc="upper right")

plt.show()
