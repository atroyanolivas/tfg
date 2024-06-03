import sys

sys.path.append("./") # Append directory where python is invoked

from config import IMG_SIZE
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator


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


train_generator = train_datagen.flow_from_directory(
    "./kaggle/Data/train_2",
    color_mode="grayscale",
    class_mode="categorical", target_size=IMG_SIZE,
    batch_size=290, shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    "./kaggle/Data/val_2",
    color_mode="grayscale",
    class_mode="categorical", target_size=IMG_SIZE,
    batch_size=25, shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    "./kaggle/Data/test_2",
    color_mode="grayscale",
    class_mode="categorical", target_size=IMG_SIZE,
    batch_size=108, shuffle=True
)



X_train, Y_train = next(train_generator)
print("X_train.shape", X_train.shape)

X_val, Y_val = next(validation_generator)
print("X_val.shape", X_val.shape)

X_test, Y_test = next(test_generator)
print("X_test.shape", X_test.shape)

# all_dataset = np.concatenate((X_train, X_val, X_test), axis=0)
# np.random.shuffle(all_dataset)
# l_train = len(X_train)
# l_val = len(X_val)
# l_test = len(X_test)

# X_train = all_dataset[:l_train, :]
# X_val = all_dataset[l_train:l_train + l_val, :]
# X_test = all_dataset[l_train + l_val:l_train + l_val + l_test, :]

np.save('./kaggle/Data/X_train.npy', X_train)
np.save('./kaggle/Data/Y_train.npy', Y_train)

np.save('./kaggle/Data/X_val.npy', X_val)
np.save('./kaggle/Data/Y_val.npy', Y_val)

np.save('./kaggle/Data/X_test.npy', X_test)
np.save('./kaggle/Data/Y_test.npy', Y_test)