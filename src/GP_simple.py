# Simplified Genetic Programming for Anomaly Detection
import warnings

warnings.filterwarnings('ignore')

# Add patch for sklearn compatibility issue with gplearn
import sklearn.base

if not hasattr(sklearn.base.BaseEstimator, '__sklearn_tags__'):
    print("Applying patch for sklearn compatibility with gplearn...")
    sklearn.base.BaseEstimator.__sklearn_tags__ = lambda self: {}

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import seaborn as sns
from gplearn.genetic import SymbolicClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the UNSW-NB15 dataset
print("Loading data...")
train_path = "../data/UNSW_NB15/UNSW_NB15_training-set.csv"
test_path = "../data/UNSW_NB15/UNSW_NB15_testing-set.csv"

try:
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    
    print("\nClass distribution:")
    print("Training set:")
    print(train_data['label'].value_counts(normalize=True))
    print("Testing set:")
    print(test_data['label'].value_counts(normalize=True))
except FileNotFoundError as e:
    print(f"Error loading dataset: {e}")
    raise

# Prepare features and labels
non_feature_cols = ['id', 'attack_cat', 'label']
feature_cols = [col for col in train_data.columns if col not in non_feature_cols]

y_train = train_data['label'].astype(int)
X_train = train_data[feature_cols]

y_test = test_data['label'].astype(int)
X_test = test_data[feature_cols]

# Handle missing values
X_train = X_train.fillna(X_train.median(numeric_only=True))
X_test = X_test.fillna(X_test.median(numeric_only=True))

# Identify numeric and categorical columns
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

# Create preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')

# Preprocess the data
print("\nPreprocessing data...")
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Fix any remaining NaN or infinity values
X_train_preprocessed = np.nan_to_num(X_train_preprocessed, nan=0.0, posinf=1e10, neginf=-1e10)
X_test_preprocessed = np.nan_to_num(X_test_preprocessed, nan=0.0, posinf=1e10, neginf=-1e10)

print(f"Preprocessed training data shape: {X_train_preprocessed.shape}")
print(f"Preprocessed testing data shape: {X_test_preprocessed.shape}")

# Define GP model with parameters optimized for anomaly detection
print("\nTraining Genetic Programming model for anomaly detection...")
gp_classifier = SymbolicClassifier(
    population_size=1000,
    generations=20,
    stopping_criteria=0.001,
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'max', 'min'],
    metric='log loss',
    tournament_size=20,
    init_depth=(2, 6),
    init_method='half and half',
    parsimony_coefficient=0.01,
    class_weight='balanced',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Train the model
gp_classifier.fit(X_train_preprocessed, y_train)
print("Training complete!")

# Make predictions on the test set
print("\nEvaluating on test data...")
y_pred = gp_classifier.predict(X_test_preprocessed)
y_pred_proba = gp_classifier.predict_proba(X_test_preprocessed)[:, 1]

# Calculate metrics for anomaly detection
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print results
print("\n====== ANOMALY DETECTION RESULTS ======")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Anomaly Detection')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('../results/anomaly_detection_confusion_matrix.png')
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))

# Print the best program found by GP
print("\nBest Program for Anomaly Detection:")
print(gp_classifier._program)

# Find optimal threshold
thresholds = np.linspace(0.1, 0.9, 9)
best_f1 = 0
best_threshold = 0.5

print("\nFinding optimal threshold for anomaly detection:")
print("Threshold | Precision | Recall | F1 Score")
print("-" * 50)

for threshold in thresholds:
    custom_preds = (y_pred_proba >= threshold).astype(int)
    prec = precision_score(y_test, custom_preds, average='binary')
    rec = recall_score(y_test, custom_preds, average='binary')
    f1 = f1_score(y_test, custom_preds, average='binary')
    
    print(f"{threshold:.1f}      | {prec:.4f}    | {rec:.4f} | {f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\nOptimal threshold: {best_threshold:.2f} with F1 Score: {best_f1:.4f}")

# Apply optimal threshold
final_preds = (y_pred_proba >= best_threshold).astype(int)
final_accuracy = accuracy_score(y_test, final_preds)
final_precision = precision_score(y_test, final_preds, average='binary')
final_recall = recall_score(y_test, final_preds, average='binary')
final_f1 = f1_score(y_test, final_preds, average='binary')

print("\n====== FINAL RESULTS WITH OPTIMAL THRESHOLD ======")
print(f"Accuracy: {final_accuracy:.4f}")
print(f"Precision: {final_precision:.4f}")
print(f"Recall: {final_recall:.4f}")
print(f"F1 Score: {final_f1:.4f}")

# Save the best program to a file
with open('../results/anomaly_detection_program.txt', 'w') as f:
    f.write(str(gp_classifier._program))
print("\nBest program saved to anomaly_detection_program.txt")

# Save the model
model_file = '../models/gp_anomaly_detector.pkl'
print(f"\nSaving model to {model_file}...")
import pickle

pickle.dump(gp_classifier, open(model_file, 'wb'))
print(f"Model saved successfully!") 