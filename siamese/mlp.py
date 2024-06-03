import sys

sys.path.append("./")

from config import SIA_BATCH_SIZE, VAL_SPLIT, IMG_SIZE
from siamese.model import instanciate_siamese_model

import tensorflow as tf
import numpy as np
import pandas as pd


# ----------------------------------------------------------------

X_val1 = np.load('./dataset/arrays/siamese/X_val1.npy')
X_val2 = np.load('./dataset/arrays/siamese/X_val2.npy')
Y_val = np.load('./dataset/arrays/siamese/Y_val.npy')

# ----------------------------------------------------------------

# Load the saved model
loaded_model = instanciate_siamese_model()
loaded_model.load_weights("./model/siamese_prueba_weights.weights.h5")

print("-------------------------------------------")
loaded_model.summary()

predictions = loaded_model.predict([X_val1, X_val2], batch_size=SIA_BATCH_SIZE)
print("predictions[:10]:\n", np.hstack((predictions, Y_val.reshape((Y_val.shape[0], 1)))))

binary_predictions = (predictions > 0.5).astype('int32')
accuracy = np.mean(np.equal(binary_predictions, Y_val))
print("Validation Accuracy:", accuracy)

exit()

# Extract the embedding layer from the loaded model
embedding_layer = loaded_model.get_layer('model')  # Replace 'model' with the actual name of your embedding layer

# Create a new model with only the embedding layer
embedding_model = tf.keras.models.Model(inputs=embedding_layer.input, outputs=embedding_layer.output)

# Summary of the embedding model
embedding_model.summary()
