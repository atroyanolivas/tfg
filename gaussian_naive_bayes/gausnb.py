import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# # Load grayscale images dataset
# X_train = np.load('./dataset/arrays/X_train_2_class.npy')
Y_train = np.load('./dataset/arrays/Y_train_2_class.npy')
Y_train = np.argmax(Y_train, axis=1)
# print("Y_train.shape =", Y_train.shape)

# X_val = np.load('./dataset/arrays/X_val_2_class.npy')
Y_val = np.load('./dataset/arrays/Y_val_2_class.npy')
Y_val = np.argmax(Y_val, axis=1)

# all_dataset = np.concatenate((X_train, X_val), axis=0)
# # all_dataset = (all_dataset - np.mean(all_dataset, axis=0)) / np.std(all_dataset, axis=0)

# l_train = len(X_train)
# l_val = len(X_val)

# def apply_PCA_to_batch(batch, n_components=179): # 335
#     original_shape = batch.shape
#     first_channel = batch[:, :, :, 0] # Separate the channels, since there is only one
#     print("first_channel.shape =", first_channel.shape)
#     observation_matrix = np.reshape(first_channel, (-1, original_shape[1] * original_shape[2]))
#     print("observation_matrix.shape =", observation_matrix.shape)

#     scaler = StandardScaler()
#     norm_obs = scaler.fit_transform(observation_matrix)

#     pca = PCA(n_components)
#     transformed_batch = pca.fit_transform(norm_obs)

#     return transformed_batch



# transformed_data = apply_PCA_to_batch(all_dataset)
# print("transformed_data.shape =", transformed_data.shape)

# X_train = transformed_data[:l_train, :]
# X_val = transformed_data[l_train:l_train + l_val, :]

X_train = np.load("./test/arrays/X_train_pca.npy")
X_val = np.load("./test/arrays/X_val_pca.npy")


# # Train Naive Bayes classifier
# clf = GaussianNB()
# clf.fit(X_train, Y_train)

# # Make predictions
# y_pred = clf.predict(X_val)

# # Evaluate accuracy
# accuracy = accuracy_score(Y_val, y_pred)
# print("Accuracy:", accuracy)

param_grid = {
    'var_smoothing': np.logspace(0, -9, num=100)
}

# Set up the grid search with accuracy as the scoring metric
grid_search = GridSearchCV(GaussianNB(), param_grid, cv=5, scoring='accuracy')

# Train the model with the grid search
grid_search.fit(X_train, Y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate accuracy
accuracy = accuracy_score(Y_val, best_model.predict(X_val))
print("Accuracy:", accuracy)


