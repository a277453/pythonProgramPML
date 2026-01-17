import pandas as pd

def load_and_explore_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Display basic information about the dataset
    print("Dataset Info:")
    df.info()

    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    print("\nMissing values in each column:")
    print(df.isnull().sum())

    print("\nUnique values in the 'Ailment' column:")
    print(df['Ailment'].unique())

    return df

if __name__ == '__main__':
    file_path = 'patient-data.csv'
    load_and_explore_data(file_path)

