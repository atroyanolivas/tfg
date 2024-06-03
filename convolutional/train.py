import sys

sys.path.append("./")

from config import LEARN_RATE, CONV_EPOCHS, BATCH_SIZE, IMG_SIZE

# from convolutional.model import model_vgg8 as model
from convolutional.model import build_conv_model
# from data.data_augmentation import train_generator, validation_generator

import numpy as np

# from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from keras_tuner.tuners import GridSearch

train_datagen = ImageDataGenerator(
    rescale=1./255
    # rescale = 1./255,
    # rotation_range = 1.8,
    # width_shift_range = 0.025,
    # height_shift_range = 0.025,
    # fill_mode = 'constant' # constan
)
val_datagen = ImageDataGenerator(
    rescale = 1./255
)

# # Create the DirectoryIterator objects for train and validation
# train_generator = train_datagen.flow_from_directory("./data/train", color_mode="grayscale", 
#                                                     batch_size=BATCH_SIZE, 
#                                                     shuffle=True, class_mode='categorical', target_size=IMG_SIZE)

# validation_generator =  val_datagen.flow_from_directory("./data/validation", color_mode="grayscale", 
#                                                         batch_size=BATCH_SIZE, 
#                                                         shuffle=True,
#                                                         class_mode='categorical', target_size=IMG_SIZE)

X_train = np.load('./dataset/arrays/X_train.npy')
# mean = np.mean(X_train, axis=0)
# std = np.std(X_train, axis=0)
# X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

Y_train = np.load('./dataset/arrays/Y_train.npy')

X_val = np.load('./dataset/arrays/X_val.npy')
# X_val = (X_val - np.mean(X_val, axis=0)) / np.std(X_val, axis=0)
Y_val = np.load('./dataset/arrays/Y_val.npy')

# initial_learning_rate = 0.1
# lr_schedule = ExponentialDecay(
#     initial_learning_rate,
#     decay_steps=10000,
#     decay_rate=0.96,
#     staircase=True)

# Compiling the model
# model.compile(
#     optimizer=Adam(learning_rate=LEARN_RATE),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )

tuner = GridSearch(
    build_conv_model,
    objective="val_accuracy",
    max_trials=150,
    directory="./convolutional/",
    project_name="conv_tuning"
)

tuner.search(X_train, Y_train, epochs=64, validation_data=(X_val, Y_val))

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

# Fit the data (perform the training)
history = best_model.fit(
    x=X_train,
    y=Y_train,
    epochs=CONV_EPOCHS,
    batch_size=BATCH_SIZE,
    
    shuffle=True,

    validation_data=(X_val, Y_val),
    validation_batch_size=BATCH_SIZE,
    verbose=1
)
# Save the model in "./model" directory
best_model.save("./model/vgg18.keras")

import matplotlib.pyplot as plt

# plt.style.use("tableau-colorblind10")
plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  

epochs = range(len(history.history["accuracy"]))

## Plot accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(epochs, history.history["accuracy"], colors[0], label='Training accuracy')
ax1.plot(epochs, history.history["val_accuracy"], colors[1], label='Validation accuracy')
# ax2.plot([0, 0], [epochs, epochs], color="white")
# ax2.plot([epochs, epochs], [0, 0], color="white")
ax1.set_title('Training and validation accuracy')
ax1.grid()
ax1.legend(loc="lower right")

## Plot Loss
ax2.plot(epochs, history.history["loss"], colors[4], label='Training Loss')
ax2.plot(epochs, history.history["val_loss"], colors[5], label='Validation Loss')
# ax2.plot([0, 0], [0, epochs], color="white")
# ax2.plot([0, epochs], [0, 0], color="white")
ax2.set_title('Training and validation Loss')
ax2.grid()
ax2.legend(loc="upper right")

plt.show()