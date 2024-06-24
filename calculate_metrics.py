import pandas as pd

# Load the predicted CSV file
predicted_file_path = 'predicted.csv'  # Replace with the correct path to your predicted.csv file
df_predicted = pd.read_csv(predicted_file_path)

# Check if the 'is_fraud' column exists
if 'is_fraud' not in df_predicted.columns:
    print("The actual label column 'is_fraud' is missing from the predicted.csv file.")
else:
    # Calculate True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
    TP = len(df_predicted[(df_predicted['is_fraud'] == 1) & (df_predicted['prediction'] == 1)])
    TN = len(df_predicted[(df_predicted['is_fraud'] == 0) & (df_predicted['prediction'] == 0)])
    FP = len(df_predicted[(df_predicted['is_fraud'] == 0) & (df_predicted['prediction'] == 1)])
    FN = len(df_predicted[(df_predicted['is_fraud'] == 1) & (df_predicted['prediction'] == 0)])
    
    # Calculate Precision, Recall, and F1 Score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision * recall) if (precision + recall) > 0 else 0
    
    print(f"Number of True Positives (TP): {TP}")
    print(f"Number of True Negatives (TN): {TN}")
    print(f"Number of False Positives (FP): {FP}")
    print(f"Number of False Negatives (FN): {FN}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
