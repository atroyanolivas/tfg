import subprocess

print("1.")
# Execute './preprocess/data_formatter2.py'
subprocess.run(["python", "./preprocess/data_formatter.py"])
print("--------------------------------------------------")

print("2.")
# Execute './preprocess/split_classes.py'
subprocess.run(["python", "./preprocess/create_dataframe.py"])
print("--------------------------------------------------")
print("3.")
# Execute './preprocess/split_classes.py'
subprocess.run(["python", "./preprocess/image_process.py"])
print("--------------------------------------------------")