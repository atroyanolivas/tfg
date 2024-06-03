from PIL import Image
import shutil
import os
import sys
import pandas as pd

sys.path.append("./")
from config import TRAIN_SPLIT, VAL_SPLIT, IMG_SIZE

df = pd.read_csv("./dataset/dataset_frame.csv")

data_val = df[0: 3 * VAL_SPLIT]
data_train = df[3 * VAL_SPLIT: 3 * TRAIN_SPLIT]

dirs = ["train", "validation"]
path = r"E:\TFG\model_src\data"

for dir, dat in zip(dirs, [data_train, data_val]): #, test]):
    # aux_path = path + dir
    aux_path = os.path.join(path, dir)

    # Check if the directory exists
    if os.path.exists(path0 := os.path.join(aux_path, "0")):
        shutil.rmtree(path0)
    if os.path.exists(path1 := os.path.join(aux_path, "1")):
        shutil.rmtree(path1)
    if os.path.exists(path2 := os.path.join(aux_path, "2")):
        shutil.rmtree(path2)

    # Create the directory
    os.makedirs(path0)
    os.makedirs(path1)
    os.makedirs(path2)
    print("Created directories: \n")
    print(path0)
    print(path1)
    print(path2)
    # to_delete = []
    for index, row in dat.iterrows():
        # Image.open(row["filename"]).resize(IMG_SIZE).save(path + dir + "/" + str(row["class"]) + "/" + row["filename"].split('/')[-1])
        # print("row[filname]: \n", row["filename"])
        try:
            Image.open(row["filename"]).resize(IMG_SIZE).save(os.path.join(path, dir, str(row["class"]), row["filename"].split("\\")[-1]))
        except:
            print("Continued, cannot open", row["filename"])
            # to_delete.append(index)
            continue