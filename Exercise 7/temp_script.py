import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Read the CSV file
df = pd.read_csv('patient-data.csv')

# Get the number of rows and columns
rows, columns = df.shape
print(f"Number of rows: {rows}")
print(f"Number of columns: {columns}")

# Find missing data
missing_data = []
for col in df.columns:
    for i, val in enumerate(df[col].isnull()):
        if val:
            missing_data.append({'row': i, 'column': col})

# Display missing data in a tabular format
if missing_data:
    print("\nMissing data:")
    missing_df = pd.DataFrame(missing_data)
    print(missing_df.to_string(index=False))
else:
    print("\nNo missing data found.")

# Descriptive Statistics
# Separate columns by level of measurement
ratio_columns = [col for col in df.columns if col.startswith('p')]
nominal_columns = ['Ailment']

# Descriptive statistics for ratio scale columns
print("Descriptive Statistics for Ratio Scale Columns:")
print(df[ratio_columns].describe())

# Descriptive statistics for nominal scale columns
print("\nDescriptive Statistics for Nominal Scale Columns:")
print(df[nominal_columns].value_counts().to_frame('Frequency'))

def find_outliers(data):
    # Calculate Q1 and Q3
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers

for col1 in ratio_columns:
    print(f" {col1} outliers : {find_outliers(df[col1])}");

# Implementing the strategy for handling missing values
# Make a copy of the dataframe to keep the original data intact
df_cleaned = df.copy()

# 1. Delete rows with missing 'Ailment' values
df_cleaned.dropna(subset=['Ailment'], inplace=True)

# 2. Delete row 773
if 773 in df_cleaned.index:
    df_cleaned.drop(index=773, inplace=True)

# 3. Impute missing values in p15 and p16 with the median
for col in ['p15', 'p16']:
    if df_cleaned[col].isnull().any():
        median_val = df_cleaned[col].median()
        df_cleaned[col].fillna(median_val, inplace=True)

# Display the info of the cleaned dataframe to verify the changes
print("Info of the cleaned dataframe:")
df_cleaned.info()

# Prepare the data
# Define features (X) and target (y)
X = df_cleaned.drop('Ailment', axis=1)
y = df_cleaned['Ailment']

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Create and train the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
log_reg.fit(X_train, y_train)

# Make predictions
y_train_pred = log_reg.predict(X_train)
y_test_pred = log_reg.predict(X_test)

# Decode the predictions to original labels for better readability in reports
y_train_pred_labels = le.inverse_transform(y_train_pred)
y_test_pred_labels = le.inverse_transform(y_test_pred)
y_train_labels = le.inverse_transform(y_train)
y_test_labels = le.inverse_transform(y_test)


# --- Train Metrics ---
print("--- Train Metrics ---")
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Accuracy: {train_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_train, y_train_pred, target_names=le.classes_))
print("\nConfusion Matrix:")
print(confusion_matrix(y_train, y_train_pred))


# --- Test Metrics ---
print("\n\n--- Test Metrics ---")
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {test_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))