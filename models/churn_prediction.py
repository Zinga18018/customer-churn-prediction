import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Load the processed dataset
try:
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df_processed = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
except FileNotFoundError:
    print("Error: 'telco_churn_processed.csv' not found. Please ensure the previous script ran successfully and the file is in the correct directory.")
    exit()

# Clean numerical columns
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
for col in numerical_cols:
    df_processed[col] = pd.to_numeric(df_processed[col].replace(' ', np.nan), errors='coerce').fillna(0)

# Feature Engineering
df_processed['MonthlyChargesPerTenure'] = df_processed['MonthlyCharges'] / (df_processed['tenure'] + 1)  # Avoid division by zero
df_processed['TotalChargesPerTenure'] = df_processed['TotalCharges'] / (df_processed['tenure'] + 1)
df_processed['HasMultipleServices'] = ((df_processed['PhoneService'] == 'Yes').astype(int) + 
                                     (df_processed['InternetService'] != 'No').astype(int) +
                                     (df_processed['StreamingTV'] == 'Yes').astype(int) +
                                     (df_processed['StreamingMovies'] == 'Yes').astype(int))
df_processed['HasSecurityServices'] = ((df_processed['OnlineSecurity'] == 'Yes').astype(int) +
                                     (df_processed['OnlineBackup'] == 'Yes').astype(int) +
                                     (df_processed['DeviceProtection'] == 'Yes').astype(int) +
                                     (df_processed['TechSupport'] == 'Yes').astype(int))

# Drop customerID
df_processed = df_processed.drop('customerID', axis=1)

# Convert categorical variables to numeric using one-hot encoding
categorical_cols = df_processed.select_dtypes(include=['object']).columns
df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)

print("Processed dataframe loaded successfully.")
print(f"Shape of loaded data: {df_processed.shape}")
print(f"Columns: {df_processed.columns.tolist()}")

# Define Features (X) and Target (y)
X = df_processed.drop('Churn_Yes', axis=1)
y = df_processed['Churn_Yes']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Define models with hyperparameter grids
models = {
    'XGBoost': {
        'model': XGBClassifier(random_state=42),
        'params': {
            'n_estimators': [200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
    },
    'LightGBM': {
        'model': LGBMClassifier(random_state=42),
        'params': {
            'n_estimators': [200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1],
            'num_leaves': [31, 63],
            'subsample': [0.8, 0.9]
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    }
}

# Train and tune models
best_models = {}
for name, model_info in models.items():
    print(f"\nTuning {name}...")
    grid_search = GridSearchCV(
        model_info['model'],
        model_info['params'],
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    grid_search.fit(X_train_balanced, y_train_balanced)
    best_models[name] = grid_search.best_estimator_
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

# Create voting classifier
voting_clf = VotingClassifier(
    estimators=[(name, model) for name, model in best_models.items()],
    voting='soft'
)

# Train voting classifier
voting_clf.fit(X_train_balanced, y_train_balanced)

# Evaluate models
results = {}
for name, model in best_models.items():
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'classification_report': classification_report(y_test, y_pred)
    }

# Evaluate voting classifier
y_pred_voting = voting_clf.predict(X_test_scaled)
y_pred_proba_voting = voting_clf.predict_proba(X_test_scaled)[:, 1]

results['Voting'] = {
    'accuracy': accuracy_score(y_test, y_pred_voting),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_voting),
    'classification_report': classification_report(y_test, y_pred_voting)
}

# Print results
print("\nModel Performance Summary:")
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Classification Report:\n{metrics['classification_report']}")

# Plot ROC curves
plt.figure(figsize=(10, 7))
for name, model in best_models.items():
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})")

# Add voting classifier ROC curve
fpr_voting, tpr_voting, _ = roc_curve(y_test, y_pred_proba_voting)
plt.plot(fpr_voting, tpr_voting, label=f"Voting (AUC = {roc_auc_score(y_test, y_pred_proba_voting):.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(True)
plt.savefig('images/roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance for the best model (XGBoost)
best_model = best_models['XGBoost']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title('Top 15 Most Important Features')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('images/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()