import pandas as pd# Load dataset
data = pd.read_csv(
    r"C:\Users\punit\OneDrive\Documents\Fake-news_detection\datasets\multimodal_train.tsv.zip",
    sep="\t",
    compression="zip"
)
# Show columns
print("All Columns:", data.columns)

# 🔹 Replace these column names with YOUR actual column names
TEXT_COLUMN = "clean_title"
LABEL_COLUMN = "2_way_label"

# Keep only required columns
data = data[[TEXT_COLUMN, LABEL_COLUMN]]

# Drop missing values
data = data.dropna()

print("\nAfter Cleaning:")
print(data.head())

# Save cleaned file
data.to_csv("cleaned_data.csv", index=False)

print("\nCleaned dataset saved successfully!")