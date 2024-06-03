"""
This file executes each part of the data transformation pipelone. In order:

    - './preprocess/data_formatter2.py' : Takes data in './dataset/' and creates 
    './dataset/formatted.csv' which contains the 'Class' attribute according to which
    data will be classified.
    
    - './preprocess/split_classes.py' : clasyfies data in './dataset' into 3 clases: 
    0, non-tumorous instances; 1, tumorous instances; 2 non-tumorous instances but not healthy
    (the have some pulmonary disease such as pneumonia). Creates the 'splitdata.csv' dataset.

    - './preprocess/split_data.py' : take the data and splits it into train set (60 %), 
    validation set (20 %) and test set (20 %)

"""

import subprocess

print("1.")
# Execute './preprocess/data_formatter2.py'
subprocess.run(["python", "./preprocess/data_formatter.py"])
print("--------------------------------------------------")

print("2.")
# Execute './preprocess/split_classes.py'
subprocess.run(["python", "./preprocess/split_classes.py"])
print("--------------------------------------------------")

print("3.")
# Execute './preprocess/split_data.py'
subprocess.run(["python", "./preprocess/split_data.py"])
print("--------------------------------------------------")