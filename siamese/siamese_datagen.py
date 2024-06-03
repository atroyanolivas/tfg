"""


"""
import sys
import tensorflow as tf
sys.path.append("./") # Append directory where python is invoked

from config import IMG_SIZE, SIA_BATCH_SIZE, SIA_EPOCHS, VAL_SPLIT, TRAIN_SPLIT
# from data.data_augmentation import train_datagen, val_datagen
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

df = pd.read_csv("./dataset/dataset_frame.csv")
data_train = df.iloc[3 * VAL_SPLIT: 3 * TRAIN_SPLIT]
data_val = df.iloc[:3 * VAL_SPLIT]

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

# train_generator = train_datagen.flow_from_directory(
#     "./data/train", color_mode="grayscale", batch_size=1, 
#     shuffle=True, class_mode='categorical', target_size=IMG_SIZE
# )
# validation_generator =  val_datagen.flow_from_directory(
#     "./data/validation", color_mode="grayscale", 
#     batch_size=1,
#     shuffle=True, class_mode='categorical', target_size=IMG_SIZE     
# )

train_generator = train_datagen.flow_from_dataframe(
    data_train, directory=None, x_col="filename",
    y_col="class", color_mode="grayscale",
    class_mode="raw", target_size=IMG_SIZE,
    batch_size=1, shuffle=True,
    validate_filenames=False
)

validation_generator = train_datagen.flow_from_dataframe(
    data_val, directory=None, x_col="filename",
    y_col="class", color_mode="grayscale",
    class_mode="raw", 
    batch_size=1, shuffle=True,
    validate_filenames=False
)

def generate_pairs(gen):
    
    while True:
    # for _ in range(3 * 405):

        im_lst1 = []
        im_lst2 = []
        labels = []

        for _ in range(0, SIA_BATCH_SIZE):

            im1, label1 = next(gen)
            im2, label2 = next(gen)
            # print("im1.shape:", im1.shape)
            # print("im2.shape:", im2.shape)

            im_lst1.append(im1)
            im_lst2.append(im2)
            labels.append(1. if np.array_equal(label1, label2) else 0.) # 0. for similar, 1. for dissimilar

        # yield [np.array(im1), np.array(im2)], np.array(labels)
        yield (np.array(im1), np.array(im2)), np.array(labels)


train_pair_generator = generate_pairs(train_generator)

validation_pair_generator = generate_pairs(validation_generator)
