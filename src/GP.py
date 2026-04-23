# Imports
import pickle
import warnings

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
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')

# Hyperparameter Configuration
# Population size: Larger populations explore more of the search space
population_size = 1000  # Balanced for good exploration vs. computation time

# Number of generations to evolve
generations = 20  # Balanced for good evolution vs. computation time

# Stopping criteria: If fitness score reaches this value, evolution stops
# Set to a very low value to always run for the full number of generations
stopping_criteria = 0.001  # Changed from None to a very small value

# Probability of crossover operation
p_crossover = 0.7  # Same as before

# Probability of subtree mutation
p_subtree_mutation = 0.1  # Reduced from 0.15 to ensure total ≤ 1.0

# Probability of hoist mutation
p_hoist_mutation = 0.05  # Same as before

# Probability of point mutation
p_point_mutation = 0.1  # Reduced from 0.15 to ensure total ≤ 1.0

# Use balanced class weights for imbalanced datasets
class_weight = 'balanced'  # Helps with imbalanced classes

# Other parameters
# Functions suitable for anomaly detection - only include valid gplearn functions
function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'max', 'min', 
                'sin', 'cos', 'tan']  # Removed invalid functions: sigmoid, inv, tanh

# Metric to use for fitness evaluation
metric = 'log loss'  # Default for classification

# Tournament selection size - influences selection pressure
tournament_size = 30  # Increased from 20 for more selective pressure

# Initial tree depth - controls initial complexity
init_depth = (2, 8)  # Increased max depth from 6 to 8

# Method to generate initial population
init_method = 'half and half'

# Controls bloat (complexity penalty) - reducing to allow more complex solutions
parsimony_coefficient = 0.005  # Reduced from 0.01

# Number of jobs for parallel processing
n_jobs = -1  # Use all available cores

# For reproducibility
random_state = 42

# Output model file
model_file = '../models/gp_anomaly_detector_enhanced.pkl'

# Data Loading & Preprocessing
# Load the UNSW-NB15 dataset
# The dataset is split into training and testing sets
train_path = "../data/UNSW_NB15/UNSW_NB15_training-set.csv"
test_path = "../data/UNSW_NB15/UNSW_NB15_testing-set.csv"

try:
    # Load training and testing data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    print(f"Training data loaded successfully with shape: {train_data.shape}")
    print(f"Testing data loaded successfully with shape: {test_data.shape}")
    
    # Get a summary of the dataset
    print("\nFirst few rows of the training data:")
    print(train_data.head())
    print("\nColumns in the dataset:")
    print(train_data.columns.tolist())
    
    # The 'label' column is the target (0 for normal, 1 for attack)
    # Also remove non-predictive columns like 'id' and 'attack_cat'
    non_feature_cols = ['id', 'attack_cat', 'label']
    feature_cols = [col for col in train_data.columns if col not in non_feature_cols]
    
    # Separate features and labels
    y_train = train_data['label']
    X_train = train_data[feature_cols]
    
    y_test = test_data['label']
    X_test = test_data[feature_cols]
    
    # Print class distribution
    print("\nClass distribution in training set:")
    print(y_train.value_counts(normalize=True))
    print("\nClass distribution in test set:")
    print(y_test.value_counts(normalize=True))
    
    # Check for and handle NaN values in the target variables
    if y_train.isna().any():
        print(f"Warning: Found {y_train.isna().sum()} NaN values in training target variable")
        # For classification, we'll drop rows with NaN targets
        nan_indices = y_train.isna()
        X_train = X_train[~nan_indices]
        y_train = y_train[~nan_indices]
        print(f"After removing NaN targets, training data shape: {X_train.shape}")

    if y_test.isna().any():
        print(f"Warning: Found {y_test.isna().sum()} NaN values in testing target variable")
        # For classification, we'll drop rows with NaN targets
        nan_indices = y_test.isna()
        X_test = X_test[~nan_indices]
        y_test = y_test[~nan_indices]
        print(f"After removing NaN targets, testing data shape: {X_test.shape}")

    # Convert target variables to appropriate format
    try:
        # Try to convert to integer (for classification)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not convert target to int: {e}")
        # If conversion fails, try to ensure they're at least float
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)
        # For classification, we need integer labels
        # Map unique values to integers
        unique_labels = np.unique(np.concatenate([y_train.values, y_test.values]))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        print(f"Mapping labels to integers: {label_map}")
        y_train = y_train.map(label_map)
        y_test = y_test.map(label_map)

    # Print updated class distribution after removing NaNs
    print("\nUpdated class distribution in training set:")
    print(y_train.value_counts(normalize=True))
    print("\nUpdated class distribution in test set:")
    print(y_test.value_counts(normalize=True))
    
    # Handle missing values for both datasets
    print("\nHandling missing values in features...")
    print(f"Missing values in training features before: {X_train.isna().sum().sum()}")
    print(f"Missing values in testing features before: {X_test.isna().sum().sum()}")
    X_train = X_train.fillna(X_train.median(numeric_only=True))
    X_test = X_test.fillna(X_test.median(numeric_only=True))
    print(f"Missing values in training features after: {X_train.isna().sum().sum()}")
    print(f"Missing values in testing features after: {X_test.isna().sum().sum()}")
    
    # Remove constant features that don't provide any information
    constant_features = [col for col in X_train.columns if X_train[col].nunique() <= 1]
    if constant_features:
        print(f"Removing {len(constant_features)} constant features: {constant_features}")
        X_train = X_train.drop(constant_features, axis=1)
        X_test = X_test.drop(constant_features, axis=1)
        
    # Clean up potential problematic values for GP
    # Replace infinity values with large numbers
    X_train = X_train.replace([np.inf, -np.inf], [1e10, -1e10])
    X_test = X_test.replace([np.inf, -np.inf], [1e10, -1e10])
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please check the column names and adjust the code accordingly")
    raise FileNotFoundError(f"Could not load dataset: {e}")

# Identify numeric and categorical columns
# This is a simple heuristic - adjust based on your dataset
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

# Create preprocessing pipelines
# We'll handle missing values and apply scaling
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Changed to sparse=False
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')  # Keep any other columns

# Apply preprocessing
print("Applying preprocessing to the data...")
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)
print(f"Shape of preprocessed training data: {X_train_preprocessed.shape}")
print(f"Shape of preprocessed testing data: {X_test_preprocessed.shape}")

# Check for any NaN or infinity values after preprocessing
if np.isnan(X_train_preprocessed).any() or np.isinf(X_train_preprocessed).any():
    print("Warning: NaN or infinity values detected after preprocessing. Replacing with zeros...")
    X_train_preprocessed = np.nan_to_num(X_train_preprocessed, nan=0.0, posinf=1e10, neginf=-1e10)

if np.isnan(X_test_preprocessed).any() or np.isinf(X_test_preprocessed).any():
    print("Warning: NaN or infinity values detected after preprocessing. Replacing with zeros...")
    X_test_preprocessed = np.nan_to_num(X_test_preprocessed, nan=0.0, posinf=1e10, neginf=-1e10)

# Now perform feature selection on the preprocessed data
print("\nPerforming feature selection on preprocessed data...")
# Select top features based on mutual information (80% of features)
k_best = int(0.8 * X_train_preprocessed.shape[1])
selector = SelectKBest(mutual_info_classif, k=k_best)

try:
    X_train_selected = selector.fit_transform(X_train_preprocessed, y_train)
    X_test_selected = selector.transform(X_test_preprocessed)
    
    print(f"Feature count after preprocessing: {X_train_preprocessed.shape[1]}")
    print(f"Feature count after selection: {X_train_selected.shape[1]}")
    
    # Use the selected features for training
    X_train_preprocessed = X_train_selected
    X_test_preprocessed = X_test_selected
except Exception as e:
    print(f"Feature selection failed: {e}")
    print("Continuing with all preprocessed features...")

# If the data is sparse, convert to dense format for GP
if scipy.sparse.issparse(X_train_preprocessed):
    X_train_preprocessed = X_train_preprocessed.toarray()
    X_test_preprocessed = X_test_preprocessed.toarray()
    print("Converted sparse matrices to dense format for GP")

# Get feature names after preprocessing (for interpretation)
# This section helps with interpreting the evolved programs
try:
    # Create generic feature names since we've done feature selection
    feature_names = [f"Feature_{i}" for i in range(X_train_preprocessed.shape[1])]
    print(f"Using generic feature names: Feature_0, Feature_1, ..., Feature_{X_train_preprocessed.shape[1]-1}")
except Exception as e:
    print(f"Warning: Could not create feature names: {e}")
    feature_names = [f"X{i}" for i in range(X_train_preprocessed.shape[1])]

print(f"Total number of features for GP model: {X_train_preprocessed.shape[1]}")

# Check data integrity before training
def check_data_integrity(X_train, y_train, X_test, y_test):
    """Check and fix data integrity issues before model training."""
    issues_found = False
    
    # 1. Check for NaN values in X and y
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        print("Warning: NaN values found in feature matrices. Replacing with zeros.")
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)
        issues_found = True
    
    # 2. Check for infinity values
    if np.isinf(X_train).any() or np.isinf(X_test).any():
        print("Warning: Infinity values found in feature matrices. Replacing with large values.")
        X_train = np.nan_to_num(X_train, posinf=1e10, neginf=-1e10)
        X_test = np.nan_to_num(X_test, posinf=1e10, neginf=-1e10)
        issues_found = True
    
    # 3. Check for NaN in target variables
    if isinstance(y_train, np.ndarray) and (np.isnan(y_train).any() or (hasattr(y_test, 'any') and np.isnan(y_test).any())):
        raise ValueError("NaN values in target variables. Cannot proceed with training.")
    
    # 4. Make sure everything is in the right format
    if scipy.sparse.issparse(X_train):
        print("Converting sparse matrix to dense for GP")
        X_train = X_train.toarray()
        X_test = X_test.toarray() if scipy.sparse.issparse(X_test) else X_test
        issues_found = True
    
    if issues_found:
        print("Data integrity issues detected and fixed.")
    else:
        print("Data integrity check passed.")
    
    return X_train, y_train, X_test, y_test

# Make sure data is ready for GP training
X_train_preprocessed, y_train, X_test_preprocessed, y_test = check_data_integrity(
    X_train_preprocessed, y_train, X_test_preprocessed, y_test
)

# GP Model Definition & Training
# Create the Genetic Programming model
gp_classifier = SymbolicClassifier(
    population_size=population_size,
    generations=generations,
    stopping_criteria=stopping_criteria,
    p_crossover=p_crossover,
    p_subtree_mutation=p_subtree_mutation,
    p_hoist_mutation=p_hoist_mutation,
    p_point_mutation=p_point_mutation,
    function_set=function_set,
    metric=metric,
    tournament_size=tournament_size,
    init_depth=init_depth,
    init_method=init_method,
    parsimony_coefficient=parsimony_coefficient,
    const_range=(-1.0, 1.0),  # Range for constants
    feature_names=feature_names,
    random_state=random_state,
    verbose=1,  # Set to 1 to see progress, 0 for silence
    class_weight=class_weight,
    n_jobs=n_jobs
)

print("Training GP classifier...")
# Train the GP model
gp_classifier.fit(X_train_preprocessed, y_train)
print("Training complete!")

# Evaluation
# Make predictions with probability scores
y_pred = gp_classifier.predict(X_test_preprocessed)
y_pred_proba = gp_classifier.predict_proba(X_test_preprocessed)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# For binary classification, also get class-specific metrics
binary_precision = precision_score(y_test, y_pred, average=None)
binary_recall = recall_score(y_test, y_pred, average=None)
binary_f1 = f1_score(y_test, y_pred, average=None)

print("\nModel Performance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Weighted Precision: {precision:.4f}")
print(f"Weighted Recall: {recall:.4f}")
print(f"Weighted F1 Score: {f1:.4f}")

print("\nClass-specific metrics:")
print(f"Precision - Class 0 (Normal): {binary_precision[0]:.4f}, Class 1 (Anomaly): {binary_precision[1]:.4f}")
print(f"Recall - Class 0 (Normal): {binary_recall[0]:.4f}, Class 1 (Anomaly): {binary_recall[1]:.4f}")
print(f"F1 Score - Class 0 (Normal): {binary_f1[0]:.4f}, Class 1 (Anomaly): {binary_f1[1]:.4f}")

# Generate and plot confusion matrix with percentages
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot both raw counts and percentages
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (Counts)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.subplot(1, 2, 2)
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (Normalized)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('../results/confusion_matrix.png')
plt.show()

# ROC Curve
if y_pred_proba.shape[1] == 2:  # Only for binary classification
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('../results/roc_curve.png')
    plt.show()
    
    # Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
    avg_precision = average_precision_score(y_test, y_pred_proba[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, lw=2, 
             label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig('../results/precision_recall_curve.png')
    plt.show()

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Results/Output
# Print the best program found by GP
print("\nBest GP Program:")
print(gp_classifier._program)

# Save the program visualization if available
try:
    import os

    from sympy import srepr, sympify
    from sympy.printing.dot import dotprint

    # Convert the program to a SymPy expression
    expr_str = str(gp_classifier._program).replace("add", "+").replace("sub", "-").replace("mul", "*").replace("div", "/")
    expr_str = expr_str.replace("sqrt", "sqrt").replace("sin", "sin").replace("cos", "cos").replace("log", "log")
    
    # Try to create a visualization of the program
    if os.system("which dot") == 0:  # Check if Graphviz is installed
        with open("../results/gp_program.dot", "w") as f:
            f.write(dotprint(sympify(expr_str, evaluate=False)))
        os.system("dot -Tpng ../results/gp_program.dot -o ../results/gp_program.png")
        print("Program visualization saved as gp_program.png")
except Exception as e:
    print(f"Could not create program visualization: {e}")

# Feature importance based on the program
if hasattr(gp_classifier, 'feature_importances_'):
    importances = gp_classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances from GP')
    plt.bar(range(min(20, len(importances))), 
            importances[indices[:20]], 
            align='center')
    
    # If feature names are available, use them for the x-axis
    if len(feature_names) > 0 and len(feature_names) == len(importances):
        top_feature_names = [feature_names[i] for i in indices[:20]]
        plt.xticks(range(min(20, len(importances))), top_feature_names, rotation=90)
    
    plt.tight_layout()
    plt.savefig('../results/feature_importances.png')
    plt.show()
    
    # Print top 10 most important features
    print("\nTop 10 most important features:")
    if len(feature_names) > 0 and len(feature_names) == len(importances):
        for i, idx in enumerate(indices[:10]):
            print(f"{i+1}. {feature_names[idx]} - Importance: {importances[idx]:.4f}")
    else:
        for i, idx in enumerate(indices[:10]):
            print(f"{i+1}. Feature {idx} - Importance: {importances[idx]:.4f}")

# Analyze program complexity
program_str = str(gp_classifier._program)
print(f"\nProgram complexity:")
print(f"Length: {len(program_str)}")
print(f"Number of operations: {program_str.count('(')}")

# Save the model
print(f"\nSaving model to {model_file}...")
pickle.dump(gp_classifier, open(model_file, 'wb'))
print(f"Model saved successfully!")

# Save the preprocessing pipeline too for easier reuse
preprocessor_file = model_file.replace('.pkl', '_preprocessor.pkl')
pickle.dump(preprocessor, open(preprocessor_file, 'wb'))
print(f"Preprocessor saved to {preprocessor_file}")

# Save the best program to a text file
with open('../models/best_program.txt', 'w') as f:
    f.write(str(gp_classifier._program))
print("Best program saved to best_program.txt")

# Add code for threshold tuning - for better anomaly detection
if y_pred_proba.shape[1] == 2:
    thresholds = np.linspace(0.1, 0.9, 9)
    
    print("\nThreshold tuning for anomaly detection:")
    print("Threshold | Precision | Recall | F1 Score | Accuracy")
    print("-" * 60)
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        # Convert probabilities to predictions using the threshold
        custom_preds = (y_pred_proba[:, 1] >= threshold).astype(int)
        
        # Calculate metrics
        prec = precision_score(y_test, custom_preds, average='binary')
        rec = recall_score(y_test, custom_preds, average='binary')
        f1 = f1_score(y_test, custom_preds, average='binary')
        acc = accuracy_score(y_test, custom_preds)
        
        print(f"{threshold:.1f}      | {prec:.4f}    | {rec:.4f} | {f1:.4f}    | {acc:.4f}")
        
        # Keep track of best F1 score
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold:.2f} with F1 Score: {best_f1:.4f}")
    
    # Save the best threshold
    with open('../models/best_threshold.txt', 'w') as f:
        f.write(str(best_threshold))
    print(f"Best threshold saved to best_threshold.txt")

print("\nTraining and evaluation complete. You can now use the saved model for anomaly detection.")
print("For deployment, use the anomaly_detector.py file which contains a simple API for detection.")
