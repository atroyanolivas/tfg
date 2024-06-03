"""


"""
import tensorflow as tf
import numpy as np
import sys

sys.path.append("./")
from config import SIA_EPOCHS, SIA_BATCH_SIZE

from siamese.model import instanciate_siamese_model_

from keras_tuner.tuners import RandomSearch


X1 = np.load('./dataset/arrays/siamese/X1_train_2_class.npy')
print("X1.shape =", X1.shape)
X2 = np.load('./dataset/arrays/siamese/X2_train_2_class.npy')
print("X2.shape =", X2.shape)
Y = np.load('./dataset/arrays/siamese/Y_train_2_class.npy')

X_val1 = np.load('./dataset/arrays/siamese/X_val1_2_class.npy')
X_val2 = np.load('./dataset/arrays/siamese/X_val2_2_class.npy')
Y_val = np.load('./dataset/arrays/siamese/Y_val_2_class.npy')


def contrastive_loss(y_true, y_pred):    
    # y_true = tf.cast(y_true, y_pred.dtype) # Should not be necessary

    # Calculate contrastive loss
    margin = 1
    # loss = y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
    loss = (1 - y_true) * tf.square(y_pred) +  y_true * tf.square(tf.maximum(margin - y_pred, 0))
    # loss = y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
    
    return tf.reduce_mean(loss)


tuner = RandomSearch(
    instanciate_siamese_model_,
    objective="val_accuracy",
    max_trials=100,
    directory="./siamese/",
    project_name="sia_tuning_2_class_random"
)

tuner.search([X1, X2], Y, epochs=45, validation_data=([X_val1, X_val2], Y_val))

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

print("Best hyperparameters:")
for hp in best_hps.values.keys():
    print(f"{hp}: {best_hps.get(hp)}")
best_model = tuner.hypermodel.build(best_hps)

best_model.summary()

history = best_model.fit(
    x=[X1, X2],
    y=Y,
    epochs=94,
    batch_size=SIA_BATCH_SIZE,
    
    shuffle=True,

    validation_data=([X_val1, X_val2], Y_val),
    validation_batch_size=SIA_BATCH_SIZE,
    verbose=1
)

# Save the model
# best_model.save("./test/models/siamese_best.keras")
# best_model.save_weights("./test/models/siamese_best_weights.weights.h5")

import matplotlib.pyplot as plt

# plt.style.use("tableau-colorblind10")
plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  

epochs = range(len(history.history["accuracy"]))

## Plot accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(epochs, history.history["accuracy"], colors[0], label='Training accuracy')
ax1.plot(epochs, history.history["val_accuracy"], colors[1], label='Validation accuracy')
ax1.grid()
ax1.set_title('Training and validation accuracy')
ax1.legend(loc="upper left")

## Plot Loss
ax2.plot(epochs, history.history["loss"], colors[4], label='Training Loss')
ax2.plot(epochs, history.history["val_loss"], colors[5], label='Validation Loss')
ax2.grid()
ax2.set_title('Training and validation Loss')
ax2.legend(loc="upper right")

plt.show()
