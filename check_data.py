import pandas as pd

data = pd.read_csv(r"C:\Users\punit\OneDrive\Documents\Fake-news_detection\datasets\multimodal_train.tsv.zip", sep="\t")

print(data.columns)
print("Columns in dataset:")

print("\nFirst 5 rows:")
print(data.head())