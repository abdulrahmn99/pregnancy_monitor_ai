import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

file_path = "CTG_Data.xlsx"
df = pd.read_excel(file_path)

def classify_risk(row):
    if (
        row['BP_Systolic'] > 140 or
        row['BP_Diastolic'] > 90 or
        row['FHR_Baseline'] < 110 or row['FHR_Baseline'] > 160 or
        row['FetalMovement'] < 4
    ):
        return 1
    else:
        return 0

df['status'] = df.apply(classify_risk, axis=1)

features = ['BP_Systolic', 'BP_Diastolic', 'MHR', 'FHR_Baseline', 'FetalMovement']
X = df[features]
y = df['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

model_path = "pregnancy_risk_model_updated.pkl"
joblib.dump(model, model_path)

accuracy, model_path
