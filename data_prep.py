import pandas as pd
from sklearn.model_selection import train_test_split

# Load in the full dataset as a pd dataframe
data = pd.read_csv('./data/all_data.csv')

# Perform the train test split
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Save train and test set
train.to_csv('./data/train.csv', index=False)
test.to_csv('./data/test.csv', index=False)
print("Train and test datasets saved locally.")