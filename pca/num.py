import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

X_train = np.load('./dataset/arrays/X_train_2_class.npy')
Y_train = np.load('./dataset/arrays/Y_train_2_class.npy')

X_val = np.load('./dataset/arrays/X_val_2_class.npy')
Y_val = np.load('./dataset/arrays/Y_val_2_class.npy')

X_test = np.load('./dataset/arrays/X_test_2_class.npy')
Y_test = np.load('./dataset/arrays/Y_test_2_class.npy')

print("X_train.shape =", X_train.shape)
print("Y_train.shape =", Y_train.shape)

print("X_val.shape =", X_val.shape)
print("Y_val.shape =", Y_val.shape)

print("X_test.shape =", X_test.shape)
print("Y_test.shape =", Y_test.shape)

all_dataset = np.concatenate((X_train, X_val, X_test), axis=0)
print("all_dataset.shape =", all_dataset.shape)
norm_dataset = (all_dataset - all_dataset.mean(axis=1, keepdims=True)) / all_dataset.std(axis=1, keepdims=True)
print("norm_dataset.shape =", norm_dataset.shape)

l_train = len(X_train)
l_val = len(X_val)
l_test = len(X_test)

# batch = (BATCH_SIZE, im_height, im_width, num_channels)
def apply_PCA_to_batch(batch, n_components=512):
    original_shape = batch.shape
    first_channel = batch[:, :, :, 0] # Separate the channels, since there is only one
    print("first_channel.shape =", first_channel.shape)
    observation_matrix = np.reshape(first_channel, (-1, original_shape[1] * original_shape[2]))
    print("observation_matrix.shape =", observation_matrix.shape)

    scaler = StandardScaler()
    norm_obs = scaler.fit_transform(observation_matrix)

    pca = PCA()
    pca.fit(norm_obs)
    explained_variance = pca.explained_variance_ratio_
    print(explained_variance[:10])
    cumulative = np.cumsum(explained_variance)

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker="o")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(explained_variance) + 1), cumulative, marker="o")
    plt.axhline(y=0.95, linestyle='--')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")

    plt.show()

    print("num_of_comp =", np.argmax(cumulative >= 0.95) + 1)

    transformed_batch = pca.fit_transform(observation_matrix) # Shape should be: (BATCH_SIZE, im_height*im_width)

    return transformed_batch



transformed_data = apply_PCA_to_batch(all_dataset)
print("transformed_data.shape =", transformed_data.shape)