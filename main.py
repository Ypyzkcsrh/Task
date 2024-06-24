import pandas as pd
import joblib
import sys

# Preprocessing function
def preprocess_data(df):
    df = df.drop(columns=['trans_date_trans_time', 'dob', 'trans_num', 'Unnamed: 0'], errors='ignore')
    df['long'] = (df['long'] + 360) % 360
    df['merch_long'] = (df['merch_long'] + 360) % 360

    # Handle categorical features by using one-hot encoding
    categorical_columns = ['category', 'state', 'job', 'gender']
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Ensure all columns are numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.factorize(df[col])[0]

    # Fill any missing values
    df = df.fillna(-999)

    return df

# Load the trained model
model = joblib.load('model.joblib')

def predict(input_csv, output_csv):
    # Load the input data
    df_input = pd.read_csv(input_csv)

    # Preprocess the data
    df_preprocessed = preprocess_data(df_input)

    # Ensure the same columns as training data
    model_columns = model.feature_names_in_
    df_preprocessed = df_preprocessed.reindex(columns=model_columns, fill_value=0)

    # Make predictions
    predictions = model.predict(df_preprocessed)

    # Output the predictions to a new CSV file
    df_input['prediction'] = predictions
    df_input.to_csv(output_csv, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 main.py <input_csv> <output_csv>")
    else:
        input_csv = sys.argv[1]
        output_csv = sys.argv[2]
        predict(input_csv, output_csv)
