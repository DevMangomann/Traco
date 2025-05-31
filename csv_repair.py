import pandas as pd

label_path = "./training/training089.csv"

label = pd.read_csv(label_path)

label[["x", "y"]] = label[["y", "x"]]

label.to_csv(label_path)
