import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data_path = r"C:\Users\HP\Documents\GitHub\ci-coursework\healthcare-dataset-stroke-data.csv"
df = pd.read_csv(data_path)

# Print basic info
print("Dataset Information:")
print(df.info())
print("\nBasic Statistics of the Dataset:")
print(df.describe())

# Check for missing and duplicate values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())
print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())

# Remove rows with missing/null values and duplicate rows
df_cleaned = df.drop_duplicates()
df_cleaned = df_cleaned.dropna()

# Save the cleaned dataset to a new file
output_path = r"C:\Users\HP\Documents\GitHub\ci-coursework\cleaned-stroke-prediction-dataset.csv"
df_cleaned.to_csv(output_path, index=False)
print(f"\nCleaned dataset saved to {output_path}")

# Visualize class imbalance (assuming the target column is named 'stroke')
plt.figure(figsize=(6, 4))
sns.countplot(data=df_cleaned, x='stroke', palette='viridis')
plt.title('Class Imbalance in Stroke Prediction Dataset')
plt.xlabel('Stroke (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()
