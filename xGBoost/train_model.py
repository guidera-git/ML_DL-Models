# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import joblib

# # === Load dataset from Excel ===
# df = pd.read_excel('data4.xlsx')

# # === Columns you specified ===
# expected_columns = [
#     'Gender', 'Academic Percentage', 'Study Stream', 'Degree Program',
#     'Analytical', 'Logical', 'Explaining', 'Creative', 'Detail-Oriented',
#     'Helping', 'Activity Preference', 'Project Preference'
# ]

# df = df[expected_columns]


# categorical_cols = ['Study Stream']
# numeric_cols = [
#     'Gender', 'Academic Percentage', 'Analytical', 'Logical', 'Explaining',
#     'Creative', 'Detail-Oriented', 'Helping', 'Activity Preference', 'Project Preference'
# ]
# target_col = 'Degree Program'

# # === Encode categorical features ===
# encoders = {}
# for col in categorical_cols:
#     df[col] = df[col].fillna('Unknown')
#     le = LabelEncoder()
#     le.fit(df[col].tolist() + ['Unknown'])
#     df[col] = le.transform(df[col])
#     encoders[col] = le

# # === Encode target column ===
# target_encoder = LabelEncoder()
# df[target_col] = target_encoder.fit_transform(df[target_col])

# # Save target encoder for later use
# joblib.dump(target_encoder, 'target_encoder.pkl')

# # === Scale numeric features ===
# scaler = StandardScaler()
# df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# # === Prepare features and target ===
# X = df[categorical_cols + numeric_cols]
# y = df[target_col]

# # === Split data ===
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # === Train model ===
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # === Evaluate model ===
# y_pred = model.predict(X_test)
# if len(y_test) > 0:
#     print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")


# # === Save model and preprocessors ===
# joblib.dump(model, 'model.pkl')
# joblib.dump(encoders, 'encoders.pkl')
# joblib.dump(scaler, 'scaler.pkl')

# # === Sample prediction ===
# sample = {
#     'Gender': 1, 
#     'Academic Percentage':83.18,
#     'Study Stream': 'Pre-Medical',
#     'Analytical': 4,
#     'Logical': 2,
#     'Explaining':4,
#     'Creative': 3,
#     'Detail-Oriented': 4,
#     'Helping': 5,
#     'Activity Preference': 2,
#     'Project Preference': 1
# }

# sample_df = pd.DataFrame([sample])

# # Load saved model, encoders, scaler, and target encoder
# model = joblib.load('model.pkl')
# encoders = joblib.load('encoders.pkl')
# scaler = joblib.load('scaler.pkl')
# target_encoder = joblib.load('target_encoder.pkl')

# # Encode categorical columns in sample
# for col in categorical_cols:
#     le = encoders[col]
#     val = sample_df.at[0, col]
#     if val not in le.classes_:
#         sample_df[col] = le.transform(['Unknown'])
#     else:
#         sample_df[col] = le.transform([val])

# # Scale numeric columns in sample
# sample_df[numeric_cols] = scaler.transform(sample_df[numeric_cols])

# # Predict
# prediction = model.predict(sample_df[categorical_cols + numeric_cols])
# predicted_program = target_encoder.inverse_transform(prediction)[0]

# print("=== Sample Prediction ===")
# print("Predicted Degree Program:", predicted_program)
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# === File paths ===
train_file = 'DataTraining.xlsx'
test_file = 'TestData.xlsx'
result_file = 'Result.xlsx'

# === Columns ===
expected_columns = [
    'Gender', 'Academic Percentage', 'Study Stream', 'Degree Program',
    'Analytical', 'Logical', 'Explaining', 'Creative',
    'Detail-Oriented', 'Helping', 'Activity Preference', 'Project Preference'
]
categorical_cols = ['Study Stream']
numeric_cols = [
    'Gender', 'Academic Percentage', 'Analytical', 'Logical', 'Explaining',
    'Creative', 'Detail-Oriented', 'Helping', 'Activity Preference', 'Project Preference'
]
target_col = 'Degree Program'

# === Load training data ===
df = pd.read_excel(train_file)[expected_columns].dropna()

# === Encode categorical input features ===
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# === Encode target ===
target_encoder = LabelEncoder()
df[target_col] = target_encoder.fit_transform(df[target_col])
joblib.dump(target_encoder, 'target_encoder.pkl')

# === Scale numeric features ===
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# === Prepare X and y ===
X_train = df[categorical_cols + numeric_cols]
y_train = df[target_col]

# === Train XGBoost model ===
model = XGBClassifier(n_estimators=100, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# === Save model and preprocessors ===
joblib.dump(model, 'model.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')

# === Load test data ===
test_df = pd.read_excel(test_file)
test_result_df = pd.read_excel(result_file)

# === Preprocess categorical columns ===
for col in categorical_cols:
    if col in test_df.columns:
        le = encoders[col]
        test_df[col] = test_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        test_df[col] = le.transform(test_df[col])

# === Scale numeric columns ===
test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

# === Predict ===
X_test = test_df[categorical_cols + numeric_cols]
y_pred = model.predict(X_test)
y_pred_labels = target_encoder.inverse_transform(y_pred)

# === Add predictions to result DataFrame ===
test_result_df['Predicted Degree'] = y_pred_labels

# === Compare predicted with actual from Result.xlsx ===
if target_col in test_result_df.columns:
    actual_degrees = test_result_df[target_col]
    valid_mask = actual_degrees.isin(target_encoder.classes_)

    if valid_mask.any():
        valid_actual = actual_degrees[valid_mask]
        valid_preds = [y for i, y in enumerate(y_pred) if valid_mask.iloc[i]]
        actual_y_encoded = target_encoder.transform(valid_actual)
        correct = (actual_y_encoded == valid_preds).sum()
        accuracy = accuracy_score(actual_y_encoded, valid_preds) * 100
        summary = f"{correct}/{len(valid_actual)} ({accuracy:.2f}%)"
    else:
        summary = "N/A (No Known Degrees)"
else:
    summary = "Degree Program column not found in Result.xlsx"

# === Add summary only to first row ===
summary_col = [''] * len(test_result_df)
summary_col[0] = summary
test_result_df['Total Correct & Accuracy'] = summary_col

# === Save result ===
try:
    test_result_df.to_excel(result_file, index=False)
    print(" Result saved to 'Result.xlsx'. Please check the file.")
except PermissionError:
    print("ERROR: Please close 'Result.xlsx' if it is open and run again.")

