import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Preprocessing function
def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop(columns=['trans_date_trans_time', 'dob', 'trans_num', 'Unnamed: 0'], errors='ignore')
    # Correct longitude values
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

# Load the dataset
dfTrain = pd.read_csv('fraudTrain.csv')  # Replace with actual path to fraudTrain.csv

# Preprocess the data
X_train = preprocess_data(dfTrain)
y_train = dfTrain['is_fraud']

# Print the data types of the columns after preprocessing
print(X_train.dtypes)

# Ensure X_train and y_train are aligned
X_train, y_train = X_train.align(y_train, axis=0, join='inner')

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib_file_path = 'model.joblib'
joblib.dump(model, joblib_file_path)

print(f"Model saved to {joblib_file_path}")
