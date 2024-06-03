import numpy as np

X_train = np.load('./dataset/arrays/X_train_2_class.npy')
Y_train = np.load('./dataset/arrays/Y_train_2_class.npy')
print("Y_train[:10] =", Y_train[:10])

X_val = np.load('./dataset/arrays/X_val_2_class.npy')
Y_val = np.load('./dataset/arrays/Y_val_2_class.npy')
print("Y_val[:10] =", Y_val[:10])