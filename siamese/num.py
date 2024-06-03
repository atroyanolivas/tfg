import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

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
print("all_dataset.shape =", all_dataset.shape)



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