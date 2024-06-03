import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Load pre-trained DenseNet121 model
densenet = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze convolutional layers (optional)
for layer in densenet.layers:
    layer.trainable = False

# Add custom classification head
x = GlobalAveragePooling2D()(densenet.output)
output_layer = Dense(num_classes, activation='softmax')(x)

# Create custom model
custom_model = Model(inputs=densenet.input, outputs=output_layer)

# Compile the model
custom_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
custom_model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))
