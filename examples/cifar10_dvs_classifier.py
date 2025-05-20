"""
Example script demonstrating how to train a classifier on CIFAR10-DVS features.

This script loads the features extracted from CIFAR10-DVS dataset and trains
several classifiers to evaluate the performance.
"""

import os
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Path to extracted features
FEATURES_PATH = "output/cifar10_dvs_features"

# Class names for CIFAR10-DVS
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def load_features():
    """
    Load extracted features and labels.
    
    Returns:
        X: Feature matrix
        y: Labels
    """
    features_path = Path(FEATURES_PATH)
    X = np.load(features_path / "features.npy")
    y = np.load(features_path / "labels.npy")
    
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Labels shape: {y.shape}")
    
    return X, y

def train_and_evaluate():
    """
    Train and evaluate different classifiers on the CIFAR10-DVS features.
    """
    # Load data
    X, y = load_features()
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define classifiers to evaluate
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(gamma='auto', probability=True, random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    
    # Train and evaluate each classifier
    results = {}
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        
        # Train the classifier
        clf.fit(X_train_scaled, y_train)
        
        # Evaluate on test set
        y_pred = clf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'y_pred': y_pred,
            'y_true': y_test,
            'classifier': clf
        }
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {name}')
        
        # Save plot
        output_dir = Path("output/cifar10_dvs_classifier")
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_dir / f"confusion_matrix_{name.replace(' ', '_').lower()}.png")
        plt.close()
    
    # Compare accuracies
    accuracies = [results[name]['accuracy'] for name in classifiers.keys()]
    plt.figure(figsize=(10, 6))
    plt.bar(classifiers.keys(), accuracies)
    plt.ylim(0, 1.0)
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.title('Classifier Performance Comparison')
    plt.savefig(output_dir / "classifier_comparison.png")
    plt.close()
    
    return results

def feature_importance():
    """
    Analyze feature importance using Random Forest.
    """
    # Load data
    X, y = load_features()
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importances
    plt.figure(figsize=(12, 6))
    plt.title('Feature Importances')
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("output/cifar10_dvs_classifier")
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / "feature_importances.png")
    plt.close()
    
    # Print top 5 features
    print("\nTop 5 most important features:")
    for i in range(5):
        print(f"Feature {indices[i]}: {importances[indices[i]]:.4f}")
    
    return importances, indices

if __name__ == "__main__":
    # Check if features exist
    if not Path(FEATURES_PATH).exists():
        print(f"Features directory not found: {FEATURES_PATH}")
        print("Please run 'python examples/cifar10_dvs_example.py extract' first")
        sys.exit(1)
    
    # Train and evaluate classifiers
    results = train_and_evaluate()
    
    # Analyze feature importance
    importances, indices = feature_importance()
    
    print("\nResults saved to output/cifar10_dvs_classifier/") 