import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "cancer patient data sets.xlsx")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

print(f"Loading dataset from {DATASET_PATH}...")
df = pd.read_excel(DATASET_PATH)

# Feature list (23 features)
feature_cols = [col for col in df.columns if col not in ['Patient Id', 'Level']]
target_col = 'Level'

print(f"Dataset shape: {df.shape}")
print(f"Features ({len(feature_cols)}): {feature_cols}")

# Prepare features and target
X = df[feature_cols]
y = df[target_col]

# Level mapping
level_order = ['Low', 'Medium', 'High']

# Preprocessing: MinMaxScaler (as done in notebook cells 55-56)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split for validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Multiple Models (including AdaBoost and K-Nearest Neighbors)
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'AdaBoost Classifier': AdaBoostClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

model_metrics = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=level_order)
    
    model_metrics[name] = {
        'accuracy': round(float(acc) * 100, 2),
        'precision': round(float(prec) * 100, 2),
        'recall': round(float(rec) * 100, 2),
        'f1_score': round(float(f1) * 100, 2),
        'confusion_matrix': cm.tolist()
    }
    print(f"Model: {name} -> Accuracy: {acc*100:.2f}%")

# Train primary production model on full dataset
primary_model = RandomForestClassifier(n_estimators=150, random_state=42)
primary_model.fit(X_scaled, y)

# Save models
joblib.dump(primary_model, os.path.join(MODELS_DIR, "model.joblib"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
print("Saved model and scaler to models/")

# Feature Importances
importances = primary_model.feature_importances_
feature_importance_list = [
    {'feature': col, 'importance': round(float(imp) * 100, 2)}
    for col, imp in zip(feature_cols, importances)
]
feature_importance_list.sort(key=lambda x: x['importance'], reverse=True)

# Generate EDA Stats for the frontend (matching notebook analyses)
level_counts = df['Level'].value_counts().to_dict()
level_percentages = {k: round(v / len(df) * 100, 2) for k, v in level_counts.items()}

# Feature statistics & distributions
feature_stats = {}
for col in feature_cols:
    feature_stats[col] = {
        'min': int(df[col].min()),
        'max': int(df[col].max()),
        'mean': round(float(df[col].mean()), 2),
        'median': float(df[col].median()),
        'std': round(float(df[col].std()), 2),
        'unique_values': sorted([int(x) for x in df[col].unique()]),
        'counts_by_value': {int(k): int(v) for k, v in df[col].value_counts().sort_index().items()},
        'mean_by_level': {
            lvl: round(float(df[df['Level'] == lvl][col].mean()), 2)
            for lvl in level_order
        }
    }

# Age vs Smoking data points for scatter plot (like cell 50-51)
scatter_age_smoking = []
for _, row in df.iterrows():
    scatter_age_smoking.append({
        'id': str(row['Patient Id']),
        'age': int(row['Age']),
        'smoking': int(row['Smoking']),
        'level': str(row['Level']),
        'gender': 'Male' if row['Gender'] == 1 else 'Female',
        'air_pollution': int(row['Air Pollution']),
        'coughing_blood': int(row['Coughing of Blood'])
    })

# Gender distribution by level
gender_level = {}
for gender_val, gender_label in [(1, 'Male'), (2, 'Female')]:
    gender_level[gender_label] = df[df['Gender'] == gender_val]['Level'].value_counts().to_dict()

# Correlation with Level (mapped to numeric: Low=1, Medium=2, High=3)
level_num = df['Level'].map({'Low': 1, 'Medium': 2, 'High': 3})
correlations = {}
for col in feature_cols:
    corr = float(df[col].corr(level_num))
    correlations[col] = round(corr, 3)

# Sorted correlation list
sorted_correlations = sorted([{'feature': k, 'correlation': v} for k, v in correlations.items()], key=lambda x: x['correlation'], reverse=True)

# Export stats.json
stats_payload = {
    'total_patients': len(df),
    'features_count': len(feature_cols),
    'level_counts': level_counts,
    'level_percentages': level_percentages,
    'level_order': level_order,
    'feature_cols': feature_cols,
    'feature_stats': feature_stats,
    'feature_importances': feature_importance_list,
    'model_metrics': model_metrics,
    'correlations': sorted_correlations,
    'scatter_age_smoking': scatter_age_smoking[:300],
    'gender_level': gender_level,
    'age_stats': {
        'min': int(df['Age'].min()),
        'max': int(df['Age'].max()),
        'mean': round(float(df['Age'].mean()), 2),
        'max_patient': df.sort_values('Age', ascending=False).iloc[0].to_dict()
    }
}

# Clean max_patient for JSON
stats_payload['age_stats']['max_patient'] = {k: int(v) if isinstance(v, (np.integer, int)) else str(v) for k, v in stats_payload['age_stats']['max_patient'].items()}

with open(os.path.join(MODELS_DIR, "stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats_payload, f, indent=2)
print("Saved stats.json to models/")

# Export all 1,000 patients for the dataset explorer
patients_list = []
for _, row in df.iterrows():
    record = {}
    for col in df.columns:
        val = row[col]
        if isinstance(val, (np.integer, int)):
            record[col] = int(val)
        elif isinstance(val, (np.floating, float)):
            record[col] = float(val)
        else:
            record[col] = str(val)
    patients_list.append(record)

with open(os.path.join(MODELS_DIR, "dataset.json"), "w", encoding="utf-8") as f:
    json.dump(patients_list, f, indent=2)
print(f"Saved dataset.json to models/ ({len(patients_list)} records)")

print("Training & Data Export Completed Successfully!")
