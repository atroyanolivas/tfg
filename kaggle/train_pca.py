import sys

sys.path.append("./")
from pca.model import build_pca_model
from config import BATCH_SIZE

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import numpy as np

from keras_tuner.tuners import RandomSearch


X_train = np.load('./kaggle/Data/X_train.npy')
Y_train = np.load('./kaggle/Data/Y_train.npy')
Y_train = (Y_train[:, 1] == 1).astype(int)

X_val = np.load('./kaggle/Data/X_val.npy')
Y_val = np.load('./kaggle/Data/Y_val.npy')
Y_val = (Y_val[:, 1] == 1).astype(int)

X_test = np.load('./kaggle/Data/X_test.npy')
Y_test = np.load('./kaggle/Data/Y_test.npy')
Y_test = (Y_test[:, 1] == 1).astype(int)

all_dataset = np.concatenate((X_train, X_val, X_test), axis=0)
# np.random.shuffle(all_dataset)

l_train = len(X_train)
l_val = len(X_val)
l_test = len(X_test)

# batch = (BATCH_SIZE, im_height, im_width, num_channels)
def apply_PCA_to_batch(batch, n_components=63): # 335
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

print("X_train.shape =", X_train.shape)
print("X_val.shape =", X_val.shape)
print("X_test.shape =", X_test.shape)


tuner = RandomSearch(
    build_pca_model,
    objective="val_accuracy",
    max_trials=120,
    directory="./kaggle/",
    project_name="pca_random1"
)

# tuner.search(X_train, Y_train, epochs=35, validation_data=(X_val, Y_val))

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
    epochs=20,
    batch_size=BATCH_SIZE,
    
    shuffle=True,

    validation_data=(X_val, Y_val),
    validation_batch_size=BATCH_SIZE,
    verbose=1
)

best_model.save("./kaggle/models/pca-mlp.keras")

