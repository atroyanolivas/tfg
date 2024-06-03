"""
This script will split the data (figured in 'splitdata.csv') into a
train set (60 %), validation set (20 %) and test set (20 %).
"""

import pandas as pd
import os
import shutil
import sys

sys.path.append("./")
from config import TRAIN_SPLIT, VAL_SPLIT
data = pd.read_csv("splitdata.csv")

print("Number of pictures: ", len(data))
train = data[VAL_SPLIT:] # The first 80 % of entries
validation = data[0:VAL_SPLIT]  # The following 10 % of entries
# test = data[round(len(data) * 0.9):] # the last 10 % of entries

dirs = ["train", "validation"] #, "test"]

path = "./data/"

for dir, dat in zip(dirs, [train, validation]): #, test]):
    aux_path = path + dir

    # Check if the directory exists
    if os.path.exists(path0 := aux_path + "/0"):
        shutil.rmtree(path0)
    if os.path.exists(path1 := aux_path + "/1"):
        shutil.rmtree(path1)
    if os.path.exists(path2 := aux_path + "/2"):
        shutil.rmtree(path2)

    # Create the directory
    os.makedirs(path0)
    os.makedirs(path1)
    os.makedirs(path2)
    for index, row in dat.iterrows():
        shutil.move("./data/2/" + (name := row["ImageID_2"]), aux_path + "/2/" + name) # Move instance 2
        shutil.move("./data/1/" + (name := row["ImageID_1"]), aux_path + "/1/" + name) # Move instance 1
        shutil.move("./data/0/" + (name := row["ImageID_0"]), aux_path + "/0/" + name) # Move instace 0
        
