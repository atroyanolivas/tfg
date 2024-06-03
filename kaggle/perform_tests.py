import sys

sys.path.append("./")

from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from kaggle.train_gausnb import best_model as gausnb

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

conv = load_model("./kaggle/models/conv-final.keras")
pca_mlp = load_model("./kaggle/models/pca-mlp-final.keras")

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

X_test_pca = transformed_data[l_train + l_val:l_train + l_val + l_test, :]

res = {"pca": dict(), "conv": dict(), "gaussnb": dict()}

res["pca"]["loss"], res["pca"]["accuracy"] = pca_mlp.evaluate(X_test_pca, Y_test)
res["conv"]["loss"], res["conv"]["accuracy"] = conv.evaluate(X_test, Y_test)

print("Train: ", conv.evaluate(X_train, Y_train))
print("Val: ", conv.evaluate(X_val, Y_val))
print("Test: ", conv.evaluate(X_test, Y_test))

res["gaussnb"]["accuracy"] = accuracy_score(Y_test, gausnb.predict(X_test_pca))
res["gaussnb"]["loss"] = None

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    # plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()



for key, model, X, Y in [("pca", pca_mlp, X_test_pca, Y_test), 
                         ("conv", conv, X_test, Y_test), ("gaussnb", gausnb, X_test_pca, Y_test)]:
    print("\n", key, "\n")

    res[key]["precision"] = precision_score(Y, pre := np.round(model.predict(X)), average='weighted')
    res[key]["recall"] = recall_score(Y, pre, average='weighted')
    res[key]["f1"] = f1_score(Y, pre, average='weighted')

    # if key == "conv": pre = np.argmax(pre, axis=1)

    cm = confusion_matrix(Y_test, pre)
    plot_confusion_matrix(cm, title=f'Confusion Matrix for {key.upper()} Model')

print(res)
