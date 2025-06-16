import pandas as pd

# Paths
label_path = "./training/training0101.csv"
merge_path = "./training/training01012.csv"

# Load CSVs
label = pd.read_csv(label_path, index_col=0)
merge = pd.read_csv(merge_path, index_col=0)

# Assign next hexbug ID to the merge file
next_hexbug_id = label['hexbug'].max() + 1
merge['hexbug'] = next_hexbug_id

# Combine the DataFrames
combined = pd.concat([label, merge], ignore_index=True)

# Sort by t (time) and hexbug ID
combined = combined.sort_values(by=['t', 'hexbug']).reset_index(drop=True)

# Save to file
combined.to_csv("./training/training0101.csv", index=True)
