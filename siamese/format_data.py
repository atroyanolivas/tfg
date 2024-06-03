import pandas as pd
import sys

sys.path.append("./")
from config import TRAIN_SPLIT, VAL_SPLIT


df = pd.read_csv("./dataset/dataset_frame.csv")

train_split = 0.86
val_split = 0.07
test_split = 0.07

df = pd.read_csv("./dataset/dataset_frame.csv")
print("df.shape =", df.shape)
shuffled_df = df.sample(frac=1, random_state=None).reset_index(drop=True)
print("shuffled_df.shape =", shuffled_df.shape)
shuffled_df.reset_index(drop=True, inplace=True)

L = len(shuffled_df)

train = shuffled_df.iloc[int(val_split * L):int((train_split + val_split) * L)]
validation = shuffled_df.iloc[0:int(val_split * L)]
test = shuffled_df.iloc[int((train_split + val_split) * L):int((train_split + val_split + test_split) * L)]

print("len(train) =", len(train))
print("len(validation) =", len(validation))

data_train = pd.DataFrame(columns=["filename", "class"])
data_val = pd.DataFrame(columns=["filename", "class"])
data_test = pd.DataFrame(columns=["filename", "class"])

j = 1
for df, res in zip([train, validation, test], [data_train, data_val, data_test]):
    rows_0 = df[df["class"] == 0].reset_index(drop=True)
    rows_1 = df[df["class"] == 1].reset_index(drop=True)
    rows_2 = df[df["class"] == 2].reset_index(drop=True)

    print("len(rows_0) =", len(rows_0))
    print("len(rows_1) =", len(rows_1))
    print("len(rows_2) =", len(rows_2))
    
    aux = False
    # for i in range(0, l := min(len(rows_0), len(rows_1)), 2):
    for i in range(0, l := min(len(rows_0), len(rows_1), len(rows_2)), 2):
        if i+1 >= l: 
            print("l =", l)
            break

        first_0 = rows_0.iloc[i]
        snd_0 = rows_0.iloc[i + 1]

        first_1 = rows_1.iloc[i]
        snd_1 = rows_1.iloc[i + 1]

        first_2 = rows_2.iloc[i]
        snd_2 = rows_2.iloc[i + 1]

        res.loc[len(res)] = first_0
        res.loc[len(res)] = snd_0
        res.loc[len(res)] = first_1
        res.loc[len(res)] = snd_1
        res.loc[len(res)] = first_2
        res.loc[len(res)] = snd_2

        # res.loc[len(res)] = first_0
        # res.loc[len(res)] = snd_1
        # res.loc[len(res)] = first_0
        # res.loc[len(res)] = snd_2
        # res.loc[len(res)] = first_1
        # res.loc[len(res)] = snd_2
        
        # For 2 classes
        # res.loc[len(res)] = first_0
        # res.loc[len(res)] = snd_1
        # res.loc[len(res)] = snd_0
        # res.loc[len(res)] = first_1

        if j == 1:
        # if aux:
            res.loc[len(res)] = first_0
            res.loc[len(res)] = snd_1
            res.loc[len(res)] = first_0
            res.loc[len(res)] = snd_2
            res.loc[len(res)] = first_1
            res.loc[len(res)] = snd_2
        #     aux = not aux
        # else:
            res.loc[len(res)] = first_1
            res.loc[len(res)] = snd_0
            res.loc[len(res)] = first_2
            res.loc[len(res)] = snd_0
            res.loc[len(res)] = first_2
            res.loc[len(res)] = snd_1
        #     aux = not aux
        else:
            res.loc[len(res)] = first_0
            res.loc[len(res)] = snd_1
            res.loc[len(res)] = first_0
            res.loc[len(res)] = snd_2
            res.loc[len(res)] = first_1
            res.loc[len(res)] = snd_2
    
    j += 1


data_train.to_csv("./siamese/data_train.csv")
data_val.to_csv("./siamese/data_val.csv")
data_test.to_csv("./siamese/data_test.csv")