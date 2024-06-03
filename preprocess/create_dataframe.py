import pandas as pd
import os

df = pd.read_csv("./dataset/formatted.csv")

# Get every row in class 0 (healthy instance)
# rows_0 = df[(df["Class"] == 0) & (df["ImageDir"] <= 18)]
rows_0 = df[df["Class"] == 0]
            
# Get every row in class 1 (not healthy instance, tumour)
# rows_1 = df[(df["Class"] == 1) & (df["ImageDir"] <= 19)]
rows_1 = df[df["Class"] == 1]

# Get every row in class 2 (not healthy but not tumour)
# rows_2 = df[(df["Class"] == 2) & (df["ImageDir"] <= 18)]
rows_2 = df[df["Class"] == 2]

df = pd.DataFrame(columns=["filename", "class"])

print(f"len(rows_0): {len(rows_0)}, len(rows_1): {len(rows_1)}, len(rows_2): {len(rows_2)}")
# print(f"len(rows_0): {len(rows_0)}, len(rows_1): {len(rows_1)}")

d1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
path_d1 = r"E:\TFG\model_src\dataset"

d2 = [10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
path_d2 = r"D:\TFG - Álvaro\data"

d3 = [25, 26, 27, 28, 29, 30, 31]
path_d3 = r"G:\TFG - Alvaro\data"

d4 = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
path_d4 = r"H:\TFG\data"

for index, (_, instance_1) in enumerate(rows_1.iterrows()):
    if index < len(rows_0): # and index < len(rows_2):
        # for instance in [instance_1, rows_0.iloc[index]]:   
        for instance in [instance_1, rows_0.iloc[index], rows_2.iloc[index]]:
            # df.loc[len(df)] = {"filename": "./dataset/" + str(instance["ImageDir"]) + "/" + instance["ImageID"],
            # path_d = path_d1 if instance["ImageDir"] in d1 else path_d2
            if instance["ImageDir"] in d1:
                path_d = path_d1
            elif instance["ImageDir"] in d2:
                path_d = path_d2
            elif instance["ImageDir"] in d3:
                path_d = path_d3
            elif instance["ImageDir"] in d4:
                path_d = path_d4
            else: break
            df.loc[len(df)] = {"filename": os.path.join(path_d, str(instance["ImageDir"]), instance["ImageID"]),
                                "class": str(instance["Class"])}

# for i in range(index, index + 2000):
#     for instance in [rows_0.iloc[index], rows_2.iloc[index]]:
#             # df.loc[len(df)] = {"filename": "./dataset/" + str(instance["ImageDir"]) + "/" + instance["ImageID"],
#             path_d = path_d1 if instance["ImageDir"] in d1 else path_d2
#             if instance["ImageDir"] in d1:
#                 path_d = path_d1
#             elif instance["ImageDir"] in d2:
#                 path_d = path_d2
#             else: break
#             df.loc[len(df)] = {"filename": os.path.join(path_d, str(instance["ImageDir"]), instance["ImageID"]),
#                                "class": str(instance["Class"])}

df.to_csv("./dataset/dataset_frame.csv")