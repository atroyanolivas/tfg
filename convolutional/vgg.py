import sys

sys.path.append("./") # Append directory where python is invoked

from config import INPUT_SHAPE, LEARN_RATE, CONV_EPOCHS, BATCH_SIZE, SEED


import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D
from tensorflow.keras.initializers import HeNormal

from tensorflow.keras.optimizers import Adam

# Load pre-trained VGG16 model
densenet = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze convolutional layers (optional)
for layer in densenet.layers[:-3]:
    layer.trainable = False

densenet.summary()

base_model_output = densenet.output
# x = Conv2D(filters=64, kernel_size=(2, 2), 
#     #   kernel_initializer=GlorotNormal(seed=None),
#     kernel_initializer=HeNormal(seed=SEED),
#     #    kernel_regularizer=regularizers.l2(0.1), 
#     activation='relu', padding="same", input_shape=INPUT_SHAPE)(base_model_output)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.MaxPool2D((2, 2))(x)
# x = Dropout(0.1)(x) # 0.035
# x = Conv2D(filters=32, kernel_size=(2, 2), 
#     #   kernel_initializer=GlorotNormal(seed=None),
#     kernel_initializer=HeNormal(seed=SEED),
#     #    kernel_regularizer=regularizers.l2(0.1), 
#     activation='relu', padding="same")(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.MaxPool2D((2, 2))(x)
# x = Dropout(0.1)(x) # 0.035
flatten_layer = Flatten()(base_model_output)
x = tf.keras.layers.Dense(512,
                              #   kernel_initializer=GlorotNormal(seed=None),
                        kernel_initializer=HeNormal(seed=SEED),
                              #   kernel_regularizer=regularizers.l2(l2_rate), 
                              activation="relu")(flatten_layer)
x = Dropout(0.3)(x)
x = tf.keras.layers.Dense(512,
                              #   kernel_initializer=GlorotNormal(seed=None),
                        kernel_initializer=HeNormal(seed=SEED),
                              #   kernel_regularizer=regularizers.l2(l2_rate), 
                              activation="relu")(x)
output_layer = Dense(3, activation='softmax')(x)  # Custom output layer

# Create custom model
model = Model(inputs=densenet.input, outputs=output_layer)
model.summary()

X_train = np.load('./dataset/arrays/X_train.npy')
Y_train = np.load('./dataset/arrays/Y_train.npy')
print("X_train.shape =", X_train.shape)
X_train = np.concatenate((X_train, X_train, X_train), axis=-1)
print("X_train.shape =", X_train.shape)
# Y_train = np.concatenate((Y_train, Y_train, Y_train), axis=-1)



X_val = np.load('./dataset/arrays/X_val.npy')
X_val = np.concatenate((X_val, X_val, X_val), axis=-1)
Y_val = np.load('./dataset/arrays/Y_val.npy')
# Y_val = np.concatenate((Y_val, Y_val, Y_val), axis=-1)

# Transfer weights from VGG16 to custom model
for i in range(len(densenet.layers)):
    model.layers[i].set_weights(densenet.layers[i].get_weights())

# Compiling the model
model.compile(
    optimizer=Adam(learning_rate=LEARN_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Fit the data (perform the training)
history = model.fit(
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
model.save("./model/vgg16.keras")
