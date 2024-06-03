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

df = pd.read_csv("./dataset/dataset_frame.csv")
shuffled_df = df.sample(frac=1, random_state=None).reset_index(drop=True)
shuffled_df.reset_index(drop=True, inplace=True)



data_train = df.iloc[3 * VAL_SPLIT: 3 * VAL_SPLIT + 3 * TRAIN_SPLIT]
data_train["class"] = data_train["class"].astype(str)

data_val = df.iloc[:3 * VAL_SPLIT]
data_val["class"] = data_val["class"].astype(str)


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


train_generator = train_datagen.flow_from_dataframe(
    data_train, directory=None, x_col="filename",
    y_col="class", color_mode="grayscale",
    class_mode="categorical", target_size=IMG_SIZE,
    batch_size=3 * TRAIN_SPLIT, shuffle=False,
    validate_filenames=False
)

validation_generator = val_datagen.flow_from_dataframe(
    data_val, directory=None, x_col="filename",
    y_col="class", color_mode="grayscale",
    class_mode="categorical", target_size=IMG_SIZE,
    batch_size=3 * VAL_SPLIT, shuffle=False,
    validate_filenames=False
)


X_train, Y_train = next(train_generator)
X_val, Y_val = next(validation_generator)

np.save('./dataset/arrays/X_train.npy', X_train)
np.save('./dataset/arrays/Y_train.npy', Y_train)

np.save('./dataset/arrays/X_val.npy', X_val)
np.save('./dataset/arrays/Y_val.npy', Y_val)

