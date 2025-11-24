import pandas as pd
import matplotlib.pyplot as plt
import data_cleaning

# Load the data
data = pd.read_csv('../data/train.csv')

# Clean the data
data = data_cleaning.clean_data(data)

# Print data attributes
print(f"Number of rows: {data.shape[0]}")
print(f"Number of columns: {data.shape[1]}")

# Count NA Comments
num_na_comments = data['comment_text'].isna().sum()
print(f"Number of NA comments: {num_na_comments}")

# Count NA Values
num_na_values = data['comment_text'].isna().sum().sum()
print(f"Number of NA values: {num_na_values}")

# Count duplicates
duplicates = data.duplicated().sum()
print(f"Number of duplicates: {duplicates}")

