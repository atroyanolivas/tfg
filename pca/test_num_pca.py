import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

X_train = np.load('./dataset/arrays/X_train.npy')
Y_train = np.load('./dataset/arrays/Y_train.npy')

X_val = np.load('./dataset/arrays/X_val.npy')
Y_val = np.load('./dataset/arrays/Y_val.npy')

all_dataset = np.concatenate((X_train, X_val), axis=0)
all_dataset = (all_dataset - np.mean(all_dataset, axis=0)) / np.std(all_dataset, axis=0)

original_shape = all_dataset.shape
print("original_shape =", original_shape)
first_channel = all_dataset[:, :, :, 0] # Separate the channels, since there is only one
observation_matrix = np.reshape(first_channel, (-1, original_shape[1] * original_shape[2]))

pca = PCA(n_components=1024)

X_train_pca = pca.fit_transform(observation_matrix)

plt.bar(range(1, 1025), pca.explained_variance_ratio_,
        alpha=0.5,
        align='center')
plt.step(range(1, 1025), np.cumsum(pca.explained_variance_ratio_),
         where='mid',
         color='red')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal Components')
plt.show()