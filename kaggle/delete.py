import numpy as np


Y_train = np.load('./kaggle/Data/Y_train.npy')
print("Y_train[:5] =", Y_train[:10])
Y_train = (Y_train[:, 1] == 1).astype(int)
print("Y_train[:5] =", Y_train[:10])


