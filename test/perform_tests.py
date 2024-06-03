import sys

sys.path.append("./")

from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from test.main import instanciate_siamese_conv
from gaussian_naive_bayes.gausnb import best_model as gausnb

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

siamese = instanciate_siamese_conv()
conv = load_model("./test/models/conv_2_class_3.keras")
sia_mlp = load_model("./test/models/sia-mlp.keras")
pca_mlp = load_model("./test/models/PCA_MLP.keras")

X_test = np.load('./dataset/arrays/X_test_2_class.npy')
X_test_pca = np.load("./test/arrays/X_test_pca.npy")
Y_test = np.load('./dataset/arrays/Y_test_2_class.npy')
Y_test_bin = (Y_test[:, 1] == 1).astype(int)

res = {"sia": dict(), "pca": dict(), "conv": dict(), "gaussnb": dict()}

res["sia"]["loss"], res["sia"]["accuracy"] = sia_mlp.evaluate(embed_X := siamese.predict(X_test), Y_test_bin)
res["pca"]["loss"], res["pca"]["accuracy"] = pca_mlp.evaluate(X_test_pca, Y_test_bin)
res["conv"]["loss"], res["conv"]["accuracy"] = conv.evaluate(X_test, Y_test_bin)

res["gaussnb"]["accuracy"] = accuracy_score(Y_test_bin, gausnb.predict(X_test_pca))
res["gaussnb"]["loss"] = None


def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    # plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()



for key, model, X, Y in [("sia", sia_mlp, embed_X, Y_test_bin), ("pca", pca_mlp, X_test_pca, Y_test_bin), 
                         ("conv", conv, X_test, Y_test_bin), ("gaussnb", gausnb, X_test_pca, Y_test_bin)]:
    print("\n", key, "\n")

    res[key]["precision"] = precision_score(Y, pre := np.round(model.predict(X)), average='weighted')
    res[key]["recall"] = recall_score(Y, pre, average='weighted')
    res[key]["f1"] = f1_score(Y, pre, average='weighted')

    # if key == "conv": pre = np.argmax(pre, axis=1)

    cm = confusion_matrix(Y_test_bin, pre)
    plot_confusion_matrix(cm, title=f'Confusion Matrix for {key.upper()} Model')

print(res)
