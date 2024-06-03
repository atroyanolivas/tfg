"""
Configurations file for the project.

"""

# --------------------- IMAGE CROP ---------------------

# UPPER = 0.3
# LOWER = 0.75
# RIGHT = 0.46
# LEFT = 0.125

LEFT = 0.20
UPPER = 0.40
RIGHT = 0.40
LOWER = 0.60

LEFT_r = 0.63
UPPER_r = 0.23
RIGHT_r = 0.83
LOWER_r = 0.43


# --------------------- GENERAL ---------------------

BATCH_SIZE = 32

# IMG_SIZE = (300, 300)
# IMG_SIZE = (64, 64) # BEST SO FAR
# IMG_SIZE = (128, 128)
IMG_SIZE = (32, 32)
# IMG_SIZE = (224, 224)

# INPUT_SHAPE = (300, 300, 1)
# INPUT_SHAPE = (128, 128, 1)
# INPUT_SHAPE = (64, 64, 1)
INPUT_SHAPE = (32, 32, 1)

VAL_SPLIT = 65
TRAIN_SPLIT = 487

# --------------------- CONVOLUTIONAL MODELS ---------------------

# LEARN_RATE = 0.00003
# LEARN_RATE = 0.0001
LEARN_RATE = 0.0007
# LEARN_RATE = 0.005

CONV_EPOCHS = 180


# --------------------- PCA - MLP ---------------------

N_COMP = 30

PCA_BATCH_SIZE = 64

PCA_LEARN_RATE = 0.001
PCA_EPOCHS = 92

# PCA_TARGET_SIZE = (64, 64)
PCA_TARGET_SIZE = IMG_SIZE


# --------------------- SIAMESE MODELS ---------------------

# SIA_LEARN_RATE = 0.000001
# SIA_LEARN_RATE = 1e-32
# SIA_LEARN_RATE = 0.00001
# SIA_LEARN_RATE = 0.0001
SIA_LEARN_RATE = 0.001
# SIA_LEARN_RATE = 0.001
# SIA_LEARN_RATE = 0.1

SIA_EPOCHS = 120

SIA_BATCH_SIZE = 128

SEED = 17 # WITH BINARY CROSSENTROPY
