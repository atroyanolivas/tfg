import sys

sys.path.append("./") # Append directory where python is invoked

from config import IMG_SIZE, SIA_BATCH_SIZE, SIA_EPOCHS, VAL_SPLIT, TRAIN_SPLIT
# from data.data_augmentation import train_datagen, val_datagen
import numpy as np
import random
import os
import tensorflow as tf
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

train_directories = ["./data/train/0/", "./data/train/1/", "./data/train/2/"]
val_directories = ["./data/validation/0/", "./data/validation/1/", "./data/validation/2/"]


def create_gen(directories):
    gen_list = [tf.keras.utils.image_dataset_from_directory(
        directory, color_mode="grayscale", batch_size=1,
        labels=None, shuffle=False, image_size=IMG_SIZE
    ).as_numpy_iterator() for directory in directories]
    while True:
        im_0 = []
        im_1 = []
        im_2 = []
        for gen, im in zip(gen_list, [im_0, im_1, im_2]):
            try:
                im.append(next(gen))
                im.append(next(gen))
            except StopIteration:
                return

        aux = list(zip(
            [im_0[0]] * 2 + [im_1[0]] * 2 + [im_2[0]] * 2, 
            [im_0[1]] + [im_1[1], im_2[1]] * 2 + [im_0[1]],
            [1., 0.] * 3
        ))
        random.shuffle(aux)
        for i1, i2, label in aux:
            # print("label:", label)
            yield i1, i2, label

def generate_pairs(dirs):

    gen = create_gen(dirs)

    while True:

        imgs1 = []
        imgs2 = []
        labels = []
        
        for _ in range(0, SIA_BATCH_SIZE):
            try:
                i1, i2, label = next(gen)
            except StopIteration:
                gen = create_gen(dirs)
                break
            
            imgs1.append(i1[0])
            imgs2.append(i2[0])
            labels.append(label)
        # print("len(labels):", len(labels))
        if len(labels) == 0: continue
        yield [np.array(imgs1), np.array(imgs2)], np.array(labels)


train_pair_generator = generate_pairs(train_directories)
validation_pair_generator = generate_pairs(val_directories)

# i = 0
# for im_list, labels in validation_pair_generator:
#     print(i)
#     print(im_list[0].shape)
#     one = 0
#     zero = 0
#     for clas in labels:
#         if clas == 0: zero += 1
#         elif clas == 1: one += 1
#     if one != zero:
#         first = im_list
#         print(labels)
#         break
#     if i == 10: break
#     i += 1
    
# gen = create_gen(val)
# for j in range(0, 300):
#     print(f"{j}.")
#     i1, i2, labels = next(gen)
#     print("Label:", labels)


# import matplotlib.pyplot as plt
# images = first

# print(images[0].shape)
# print(images[1].shape)
# # print(labels)

# # Create figure and axes
# fig, axes = plt.subplots(1, 2)

# # Display first image
# axes[0].imshow(images[0][0])
# axes[0].axis('off')  # Turn off axis
# axes[0].set_title(f'Label: {labels[0]}')

# # Display second image
# axes[1].imshow(images[1][0])
# axes[1].axis('off')  # Turn off axis
# axes[1].set_title(f'Label: {labels[0]}')

# # Adjust layout
# plt.tight_layout()

# # Show plot
# plt.show()

# images, labels = snd

# print(images[0].shape)
# print(images[1].shape)
# print(labels)

# # Create figure and axes
# fig, axes = plt.subplots(1, 2)

# # Display first image
# axes[0].imshow(images[0][0])
# axes[0].axis('off')  # Turn off axis
# axes[0].set_title(f'Label: {labels[0]}')

# # Display second image
# axes[1].imshow(images[1][0])
# axes[1].axis('off')  # Turn off axis
# axes[1].set_title(f'Label: {labels[0]}')

# # Adjust layout
# plt.tight_layout()

# # Show plot
# plt.show()
