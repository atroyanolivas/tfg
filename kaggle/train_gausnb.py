import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

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


param_grid = {
    'var_smoothing': np.logspace(0, -9, num=100)
}

grid_search = GridSearchCV(GaussianNB(), param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, Y_train)

best_model = grid_search.best_estimator_

accuracy = accuracy_score(Y_val, best_model.predict(X_val))
print("Accuracy:", accuracy)


