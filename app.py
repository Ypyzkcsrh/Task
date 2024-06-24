from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.joblib')

def preprocess_data(df):
    required_columns = [
        'category', 'state', 'job', 'gender', 'cc_num', 'merchant', 'amt', 'first', 'last',
        'street', 'city', 'zip', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long'
    ]
    
    # Add any missing columns with default values
    for col in required_columns:
        if col not in df.columns:
            if col in ['lat', 'long', 'merch_lat', 'merch_long', 'amt', 'city_pop', 'unix_time']:
                df[col] = 0.0
            else:
                df[col] = 'unknown'

    df = df.drop(columns=['trans_date_trans_time', 'dob', 'trans_num', 'Unnamed: 0'], errors='ignore')
    df['long'] = (df['long'] + 360) % 360
    df['merch_long'] = (df['merch_long'] + 360) % 360

    categorical_columns = ['category', 'state', 'job', 'gender']
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.factorize(df[col])[0]

    df = df.fillna(-999)

    return df

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df_input = pd.DataFrame(data)
    df_preprocessed = preprocess_data(df_input)

    model_columns = model.feature_names_in_
    df_preprocessed = df_preprocessed.reindex(columns=model_columns, fill_value=0)

    predictions = model.predict(df_preprocessed)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
