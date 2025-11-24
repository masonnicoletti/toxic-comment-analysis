import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('../data/train.csv')

# Print data attributes
print(f"Number of rows: {data.shape[0]}")
print(f"Number of columns: {data.shape[1]}")

# Count NA Comments
num_na_comments = data['comment_text'].isna().sum()
print(f"Number of NA comments: {num_na_comments}")

# Count duplicates
duplicates = data.duplicated().sum()
print(f"Number of duplicates: {duplicates}")

# Print NA values
na_table = data.isna().sum()
print(f"NA Table:\n {na_table}")
