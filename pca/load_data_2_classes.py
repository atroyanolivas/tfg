"""
This script takes the .csv "dataset_frame.csv" in directory ./dataset
and encodes the data in a numpy array.
"""

import sys

sys.path.append("./") # Append directory where python is invoked

from config import IMG_SIZE, SIA_BATCH_SIZE, SIA_EPOCHS, VAL_SPLIT, TRAIN_SPLIT
import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_split = 0.86
val_split = 0.07
test_split = 0.07

df = pd.read_csv("./dataset/dataset_frame.csv")
print("df.shape =", df.shape)
shuffled_df = df.sample(frac=1, random_state=None).reset_index(drop=True)
print("shuffled_df.shape =", shuffled_df.shape)
shuffled_df.reset_index(drop=True, inplace=True)

L = len(shuffled_df)
print("L =", L)

data_train = shuffled_df.iloc[int(val_split * L): int((train_split + val_split) * L)]
data_train = data_train[data_train["class"] != 2]
data_train["class"] = data_train["class"].astype(str)


data_val = shuffled_df.iloc[:int(val_split * L)]
data_val = data_val[data_val["class"] != 2]
data_val["class"] = data_val["class"].astype(str)

data_test = shuffled_df.iloc[int((train_split + val_split) * L):int((train_split + val_split + test_split) * L)]
data_test = data_test[data_test["class"] != 2]
test_split = len(data_test)
data_test["class"] = data_test["class"].astype(str)


train_datagen = ImageDataGenerator(
    rescale= 1./ 255
    # rescale = 1./255,
    # rotation_range = 1.8,
    # width_shift_range = 0.025,
    # height_shift_range = 0.025,
    # fill_mode = 'constant' # constan
)

val_datagen = ImageDataGenerator(
    rescale = 1./255
)

test_datagen = ImageDataGenerator(
    rescale = 1./255
)


train_generator = train_datagen.flow_from_dataframe(
    data_train, directory=None, x_col="filename",
    y_col="class", color_mode="grayscale",
    class_mode="categorical", target_size=(32, 32),
    batch_size=3 * TRAIN_SPLIT, shuffle=False,
    validate_filenames=False
)

validation_generator = val_datagen.flow_from_dataframe(
    data_val, directory=None, x_col="filename",
    y_col="class", color_mode="grayscale",
    class_mode="categorical", target_size=(32, 32),
    batch_size=3 * VAL_SPLIT, shuffle=False,
    validate_filenames=False
)

test_generator = test_datagen.flow_from_dataframe(
    data_test, directory=None, x_col="filename",
    y_col="class", color_mode="grayscale",
    class_mode="categorical", target_size=(32, 32),
    batch_size=test_split, shuffle=False,
    validate_filenames=False
)


X_train, Y_train = next(train_generator)
X_val, Y_val = next(validation_generator)
X_test, Y_test = next(test_generator)

print("X_train.shape =", X_train.shape)
print("X_val.shape =", X_val.shape)
print("X_test.shape =", X_test.shape)

np.save('./dataset/arrays/X_train_2_class.npy', X_train)
np.save('./dataset/arrays/Y_train_2_class.npy', Y_train)

np.save('./dataset/arrays/X_val_2_class.npy', X_val)
np.save('./dataset/arrays/Y_val_2_class.npy', Y_val)

np.save('./dataset/arrays/X_test_2_class.npy', X_test)
np.save('./dataset/arrays/Y_test_2_class.npy', Y_test)

