import sys

sys.path.append("./")

from config import LEARN_RATE, CONV_EPOCHS, BATCH_SIZE, IMG_SIZE
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
import numpy as np

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

# Create the DirectoryIterator objects for train and validation
train_generator = train_datagen.flow_from_directory("./data/train", color_mode="grayscale", 
                                                    batch_size=BATCH_SIZE, 
                                                    shuffle=True, class_mode='categorical', target_size=IMG_SIZE)

validation_generator =  val_datagen.flow_from_directory("./data/validation", color_mode="grayscale", 
                                                        batch_size=BATCH_SIZE, 
                                                        shuffle=True,
                                                        class_mode='categorical', target_size=IMG_SIZE)

# batch = (BATCH_SIZE, im_height, im_width, num_channels)
def apply_PCA_to_batch(batch, n_components=30):
    original_shape = batch.shape
    first_channel = batch[:, :, :, 0] # Separate the channels, since there is only one
    observation_matrix = np.reshape(first_channel, (-1, original_shape[0] * original_shape[1]))

    pca = PCA(n_components)
    transformed_batch = pca.fit_transform(observation_matrix) # Shape should be: (BATCH_SIZE, im_height*im_width)

    return transformed_batch



next_batch, labels = next(train_generator)

print("next_batch.shape =", next_batch.shape)
transformed_batch = apply_PCA_to_batch(next_batch)
print("transformed_batch.shape =", transformed_batch.shape)

