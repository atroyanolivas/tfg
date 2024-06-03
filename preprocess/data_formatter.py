"""




"""

import pandas as pd

df = pd.read_csv("./dataset/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv")

# print("df:")
# print(df)

# df.info()

"""
Vamos a limpiar aquellas instancias que no estén en posición frontal y
aquellas que no estén sanas o tengan un nódulo cancerígeno.
"""

# Lo primero que vamos a hacer va a ser quitar todas las imágenes que no sean frontales.

df = df[df["ViewPosition_DICOM"] == "POSTEROANTERIOR"]

# Ahora quitaremos las instancias que, de padecer alguna enfermedad, no tengan nódulo cancerígeno
# (es decir, queremos dejar las instancias sanas y las que tienen un tumor)

def correct_strings(st):
    print("Element:", st, "|", type(st))
    res = st
    if st[0] != "\"": res = "\"" + res
    if st[-1] != "\"": res = res + "\""
    return res


df = df.dropna(subset="Labels")
print(df["Labels"])

# to_filter = set(["pseudonodule", "nodule", "multiple nodules"])
to_filter = set(["nodule", "multiple nodules"])

new = df.loc[:, ["ImageID", "ImageDir", "Labels"]]
new.info()

def string_to_list(st: str) -> list:
    """
    "['label1', 'label2', ..., 'labelN']"
    """
    res = list()
    in_word = False
    word = ""
    for i in range(0, len(st)):
        if st[i] == "\'":
            if not in_word:
                word = ""
                in_word = True
                continue
            else: # in_word == True
                res.append(word)
                in_word = False
        word += st[i]
    return res


def label_classes(x: str):
    lst_x = string_to_list(x)
    if 'normal' in lst_x: return 0
    elif any(el in to_filter for el in lst_x): return 1
    else: return 2

new["Class"] = new["Labels"].apply(label_classes)

new.to_csv("./dataset/formatted.csv")


