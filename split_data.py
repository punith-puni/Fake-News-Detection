import pandas as pd
from sklearn.model_selection import train_test_split

# Load cleaned dataset
data = pd.read_csv("cleaned_data.csv")

# Separate features and labels
X = data.iloc[:, 0]   # Text column
y = data.iloc[:, 1]   # Label column

# Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save files
train_df = pd.DataFrame({"text": X_train, "label": y_train})
test_df = pd.DataFrame({"text": X_test, "label": y_test})

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Train and Test files created successfully!")