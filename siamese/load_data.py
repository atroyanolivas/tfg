import sys

sys.path.append("./") # Append directory where python is invoked

from config import IMG_SIZE, SIA_BATCH_SIZE, SIA_EPOCHS, VAL_SPLIT, TRAIN_SPLIT
import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_train = pd.read_csv("./siamese/data_train.csv")
data_train = data_train[data_train["class"] != 2]
data_train["class"] = data_train["class"].astype(str)

data_val = pd.read_csv("./siamese/data_val.csv")
data_val = data_val[data_val["class"] != 2]
data_val["class"] = data_val["class"].astype(str)

data_test = pd.read_csv("./siamese/data_test.csv")
data_test = data_test[data_test["class"] != 2]
data_test["class"] = data_test["class"].astype(str)

len_train = len(data_train)
len_val = len(data_val)
len_test = len(data_test)

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
    class_mode="raw", target_size=IMG_SIZE,
    batch_size=1, shuffle=False,
    validate_filenames=False
)

validation_generator = val_datagen.flow_from_dataframe(
    data_val, directory=None, x_col="filename",
    y_col="class", color_mode="grayscale",
    class_mode="raw", target_size=IMG_SIZE,
    batch_size=1, shuffle=False,
    validate_filenames=False
)

test_generator = test_datagen.flow_from_dataframe(
    data_test, directory=None, x_col="filename",
    y_col="class", color_mode="grayscale",
    class_mode="raw", target_size=IMG_SIZE,
    batch_size=1, shuffle=False,
    validate_filenames=False
)

def generate_pairs(gen, length):

    im_lst1 = []
    im_lst2 = []
    labels = []

    progbar = tf.keras.utils.Progbar(length // 2)

    for i in range(0, length // 2):

        im1, label1 = next(gen)
        im2, label2 = next(gen)
        # print("im1.shape:", im1.shape)
        # print("im2.shape:", im2.shape)

        im_lst1.append(im1[0])
        im_lst2.append(im2[0])
        labels.append(0. if np.array_equal(label1, label2) else 1.)
        progbar.update(i + 1)

    # yield [np.array(im1), np.array(im2)], np.array(labels)
    return (np.array(im_lst1), np.array(im_lst2)), np.array(labels)

(X1, X2), Y = generate_pairs(train_generator, len_train) # SHUFFLE ??
np.save('./dataset/arrays/siamese/X1_train_2_class.npy', X1)
np.save('./dataset/arrays/siamese/X2_train_2_class.npy', X2)
np.save('./dataset/arrays/siamese/Y_train_2_class.npy', Y)

(X_val1, X_val2), Y_val = generate_pairs(validation_generator, len_val)
np.save('./dataset/arrays/siamese/X_val1_2_class.npy', X_val1)
np.save('./dataset/arrays/siamese/X_val2_2_class.npy', X_val2)
np.save('./dataset/arrays/siamese/Y_val_2_class.npy', Y_val)

(X_test1, X_test2), Y_test = generate_pairs(test_generator, len_test)
np.save('./dataset/arrays/siamese/X_test1_2_class.npy', X_test1)
np.save('./dataset/arrays/siamese/X_test2_2_class.npy', X_test2)
np.save('./dataset/arrays/siamese/Y_test_2_class.npy', Y_test)

print("X1.shape: ", X1.shape)
print("X2.shape: ", X2.shape)
print("Y.shape: ", Y.shape)

print("X_val1.shape: ", X_val1.shape)
print("X_val2.shape: ", X_val2.shape)
print("Y_val.shape: ", Y_val.shape)

print("X_test1.shape: ", X_test1.shape)
print("X_test2.shape: ", X_test2.shape)
print("Y_test.shape: ", Y_test.shape)
