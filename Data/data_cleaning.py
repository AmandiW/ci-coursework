import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

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

# FIXING CLASS IMBALANCE
# Load the cleaned dataset
df_cleaned = pd.read_csv("C:/Users/HP/Documents/GitHub/ci-coursework/healthcare-dataset-stroke-data.csv")

# Check dataset info
print(f"Dataset info:\n{df_cleaned.info()}")
print(f"Dataset summary:\n{df_cleaned.describe()}")
print(f"Class distribution:\n{df_cleaned['stroke'].value_counts()}")

# Data Cleaning - Remove rows with missing values or duplicates
df_cleaned.dropna(inplace=True)  # Drop rows with missing values
df_cleaned.drop_duplicates(inplace=True)  # Remove duplicates

# Encode categorical columns
label_encoder = LabelEncoder()

# Encode binary columns (gender, ever_married)
df_cleaned['gender'] = label_encoder.fit_transform(df_cleaned['gender'])  # Male = 0, Female = 1
df_cleaned['ever_married'] = label_encoder.fit_transform(df_cleaned['ever_married'])  # Yes = 1, No = 0

# Encode multi-class columns using pd.get_dummies (work_type, Residence_type, smoking_status)
df_cleaned = pd.get_dummies(df_cleaned, columns=['work_type', 'Residence_type', 'smoking_status'], drop_first=True)

# Separate features and target variable
X = df_cleaned.drop('stroke', axis=1)  # Features (all columns except 'stroke')
y = df_cleaned['stroke']  # Target variable ('stroke')

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check the new class distribution
print(f"Class distribution after SMOTE:\n{y_resampled.value_counts()}")

# Visualize class balance after SMOTE
plt.figure(figsize=(6, 4))
sns.countplot(data=pd.DataFrame({'stroke': y_resampled}), x='stroke', palette='viridis')
plt.title('Class Balance After SMOTE')
plt.xlabel('Stroke (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Combine the resampled features and target variable into a new DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)  # Create DataFrame for features
df_resampled['stroke'] = y_resampled  # Add the target variable (stroke) to the DataFrame

# Save the resampled dataset to a new CSV file
df_resampled.to_csv("C:/Users/HP/Documents/GitHub/ci-coursework/cleaned-stroke-prediction-dataset-balanced.csv", index=False)

print("Resampled and balanced dataset saved as 'cleaned-stroke-prediction-dataset-balanced.csv'")