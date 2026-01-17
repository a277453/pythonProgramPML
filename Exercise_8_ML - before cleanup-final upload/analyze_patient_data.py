import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

def analyze_patient_data(file_path):
    # 1. Read the data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # 2. Handle missing values
    # Drop rows with any missing values for simplicity.
    # A more advanced approach could be imputation.
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)


    # 3. Encode the categorical target variable 'Ailment'
    le = LabelEncoder()
    df['Ailment'] = le.fit_transform(df['Ailment'])

    # Separate features (X) and target (y)
    X = df.drop('Ailment', axis=1)
    y = df['Ailment']

    # 4. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 5. Build and train a classification model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 6. Make predictions on the test set
    y_pred = model.predict(X_test)

    # 7. Calculate and print classification metrics
    print("Classification Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision (macro): {precision_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Recall (macro): {recall_score(y_test, y_pred, average='macro'):.4f}")
    print(f"F1-score (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
    
    print(f"\nPrecision (micro): {precision_score(y_test, y_pred, average='micro'):.4f}")
    print(f"Recall (micro): {recall_score(y_test, y_pred, average='micro'):.4f}")
    print(f"F1-score (micro): {f1_score(y_test, y_pred, average='micro'):.4f}")
    
    print("\nNote: 'macro' calculates metrics for each label, and finds their unweighted mean. 'micro' calculates metrics globally by counting the total true positives, false negatives and false positives.")


if __name__ == '__main__':
    file_path = 'patient-data.csv'
    analyze_patient_data(file_path)
