import sys

sys.path.append("./")

from config import IMG_SIZE, UPPER, LOWER, LEFT, RIGHT, UPPER_r, LOWER_r, RIGHT_r, LEFT_r

from PIL import Image
import cv2
import shutil
import os
import pandas as pd
import numpy as np

df = pd.read_csv("./dataset/formatted.csv")

# Get every row in class 0 (healthy instance)
rows_0 = df[df["Class"] == 0]

# Get every row in class 1 (not healthy instance)
rows_1 = df[df["Class"] == 1]

# Get every row in class 2 (not healthy but not tumour)
rows_2 = df[df["Class"] == 2]

df = pd.DataFrame(columns=["ImageID_0", "ImageID_1", "ImageID_2"])

# Check if the directory exists
if os.path.exists(path0 := "./data/0"):
    shutil.rmtree(path0)
if os.path.exists(path1 := "./data/1"):
    shutil.rmtree(path1)
if os.path.exists(path2 := "./data/2"):
    shutil.rmtree(path2)

# Create the directory
os.makedirs(path0)
os.makedirs(path1)
os.makedirs(path2)

def get_2_crops(image: Image, factor=1.5) -> tuple:
    cropped_image_left = image.crop((image.size[0] * LEFT,
                                    image.size[1] * UPPER, image.size[0] * RIGHT,
                                    image.size[1] * LOWER))
    cropped_image_right = image.crop((image.size[0] * LEFT_r,
                                    image.size[1] * UPPER_r, image.size[0] * RIGHT_r,
                                    image.size[1] * LOWER_r))
    return cropped_image_left, cropped_image_right
    # return Image.fromarray(np.array(clahe.apply(np.uint8(cv2.normalize(cropped_image_left, None, 0, 255, cv2.NORM_MINMAX))))), \
    #     Image.fromarray(np.array(clahe.apply(np.uint8(cv2.normalize(cropped_image_right, None, 0, 255, cv2.NORM_MINMAX)))))

# def divide_image(image, rows=3, cols=3):

#     img_width, img_height = image.size
#     subimg_width = img_width // cols
#     subimg_height = img_height // rows

#     subimages = []
#     for y in range(rows):
#         for x in range(cols):
#             left = x * subimg_width
#             upper = y * subimg_height
#             right = left + subimg_width
#             lower = upper + subimg_height
#             subimage = image.crop((left, upper, right, lower))
#             subimages.append(subimage)

#     return subimages

# As there are less instances from class 1, iterate them:
# i = 0
for index, instance_1 in rows_1.iterrows():
    if index == 0: continue

    try:
        im1 = Image.open("./dataset/" + str(instance_1["ImageDir"]) + "/" + instance_1["ImageID"])
    
    except FileNotFoundError:
        print("File not found, directory " + str(instance_1["ImageDir"]))
        print("Everything OK, finishing")
        break
    im0 = Image.open("./dataset/" + str(rows_0.iloc[0]["ImageDir"]) + "/" + (notumour0 := rows_0.iloc[0]["ImageID"])) # healthy
    rows_0 = rows_0[1:].reset_index(drop=True)

    im2 = Image.open("./dataset/" + str(rows_2.iloc[0]["ImageDir"]) + "/" + (notumour2 := rows_2.iloc[0]["ImageID"])) # not tumour but not healthy
    rows_2 = rows_2[1:].reset_index(drop=True)

    keys = {0: "./data/0/" + notumour0.split('.')[0],
            1: "./data/1/" + instance_1["ImageID"].split('.')[0],
            2: "./data/2/" + notumour2.split('.')[0]}
    
    for index, im in enumerate([im0, im1, im2]):
        # im_left, im_right = get_2_crops(im)
        # # sub_im = divide_image(im_left)
        # im_left.resize(IMG_SIZE).save(keys[index] + "_left.png")
        # im_right.resize(IMG_SIZE).save(keys[index] + "_right.png")
        # for i, image in enumerate(sub_im):
        #     image.resize(IMG_SIZE).save(keys[index] + f"_left{i}.png")

    # for i in range(0, len(sub_im)):
    #     df.loc[len(df)] = {"ImageID_0": notumour0.split('.')[0] + f"_left{i}.png",
    #                 "ImageID_1": instance_1["ImageID"].split('.')[0] + f"_left{i}.png",
    #                 "ImageID_2": notumour2.split('.')[0] + f"_left{i}.png"}

    # df.loc[len(df)] = {"ImageID_0": notumour0.split('.')[0] + "_left.png",
    #                    "ImageID_1": instance_1["ImageID"].split('.')[0] + "_left.png",
    #                    "ImageID_2": notumour2.split('.')[0] + "_left.png"}
    # df.loc[len(df)] = {"ImageID_0": notumour0.split('.')[0] + "_right.png",
    #                    "ImageID_1": instance_1["ImageID"].split('.')[0] + "_right.png",
    #                    "ImageID_2": notumour2.split('.')[0] + "_right.png"}
    
df.to_csv("splitdata.csv")
