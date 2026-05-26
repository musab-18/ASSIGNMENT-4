#!/usr/bin/env python3
"""
Image Classification Pipeline using PCA (Eigenfaces) and Support Vector Machines (SVM).
Created for Machine Learning Assignment #4.

This script implements:
1. Data Loading & Exploration: Labeled Faces in the Wild (LFW) dataset.
2. Preprocessing: Stratified Train/Test Split and Feature Scaling (Standardization).
3. Dimensionality Reduction: Principal Component Analysis (PCA) to extract Eigenfaces.
4. Hyperparameter Tuning: GridSearchCV to find optimal C and gamma for an RBF SVM.
5. Model Evaluation: Accuracy, Classification Report, and Heatmapped Confusion Matrix.
6. Visual Diagnostics: Plotting test predictions with correct/incorrect color-coding.

Author: Antigravity AI
Date: May 2026
"""

import os
import ssl
# Bypass SSL certificate verification for scikit-learn dataset download on macOS
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set visualization style to a clean, premium modern design
sns.set_theme(style="white", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "font.family": "sans-serif"
})


def load_and_explore_data():
    """
    Loads the Labeled Faces in the Wild (LFW) dataset.
    Extracts faces of people who have at least 70 images to ensure sufficient training data.
    """
    print("[Step 1] Loading Labeled Faces in the Wild (LFW) dataset...")
    # download_if_missing=True will automatically download the dataset if not present locally (~200MB)
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4, download_if_missing=True)
    
    # Extract metadata and dimensions
    n_samples, h, w = lfw_people.images.shape
    X = lfw_people.data
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]
    
    print("\n--- Dataset Summary ---")
    print(f"Total number of samples: {n_samples}")
    print(f"Image dimensions: {h}x{w} pixels = {X.shape[1]} features")
    print(f"Number of classes (people): {n_classes}")
    print("Class distribution:")
    for i, name in enumerate(target_names):
        count = np.sum(y == i)
        print(f"  - {name}: {count} images ({count/n_samples*100:.1f}%)")
    print("-----------------------\n")
    
    # Plot some sample faces to explore the dataset
    fig, axes = plt.subplots(3, 5, figsize=(12, 8))
    fig.suptitle("Sample Images from LFW Dataset", weight="bold", y=0.98)
    for i, ax in enumerate(axes.flat):
        ax.imshow(lfw_people.images[i], cmap="gray")
        ax.set_title(target_names[y[i]], fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("lfw_samples.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved sample images plot to 'lfw_samples.png'")
    
    return X, y, target_names, h, w


def preprocess_data(X, y):
    """
    Splits the dataset into training and testing sets, keeping the class ratio balanced,
    and applies standard scaling to features (essential for PCA!).
    """
    print("[Step 2] Splitting and scaling dataset...")
    # Stratified split ensures that class proportions are preserved in both train and test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Feature Scaling: PCA is highly sensitive to variance. We must scale features to mean=0, std=1.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def apply_pca(X_train, X_test, h, w, n_components=150):
    """
    Applies Principal Component Analysis (PCA) to reduce the dimensionality of the dataset.
    Plots cumulative explained variance and the first few 'eigenfaces' (principal components).
    """
    print(f"[Step 3] Applying PCA (Dimensionality Reduction to {n_components} components)...")
    pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True, random_state=42).fit(X_train)
    
    # Transform both train and test sets to the lower-dimensional eigenspace
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print(f"Original feature size: {X_train.shape[1]}")
    print(f"Reduced feature size: {X_train_pca.shape[1]}")
    
    # 1. Plot Explained Variance
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_components + 1), explained_variance, marker='o', linestyle='-', color='#4A90E2', markevery=10)
    plt.axhline(y=0.90, color='r', linestyle='--', alpha=0.7, label='90% Explained Variance')
    plt.title("Cumulative Explained Variance Ratio vs. Number of Components", weight="bold")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("explained_variance.png", dpi=300)
    plt.close()
    print(f"Total explained variance by {n_components} components: {explained_variance[-1]*100:.2f}%")
    print("Saved cumulative explained variance plot to 'explained_variance.png'")
    
    # 2. Visualize Eigenfaces (First 15 Principal Components reshaped as images)
    eigenfaces = pca.components_.reshape((n_components, h, w))
    fig, axes = plt.subplots(3, 5, figsize=(12, 8))
    fig.suptitle("Eigenfaces: First 15 Principal Components", weight="bold", y=0.98)
    for i, ax in enumerate(axes.flat):
        ax.imshow(eigenfaces[i], cmap="plasma")  # plasma colormap highlights differences in lighting/features
        ax.set_title(f"PC {i+1}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("eigenfaces.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved first 15 Eigenfaces plot to 'eigenfaces.png'\n")
    
    return X_train_pca, X_test_pca, pca


def train_svm_grid_search(X_train_pca, y_train):
    """
    Trains a Support Vector Machine (SVM) classifier with Radial Basis Function (RBF) kernel.
    Uses 5-fold cross-validation grid search to tune hyperparameters C and gamma.
    """
    print("[Step 4] Training SVM classifier and optimizing hyperparameters...")
    param_grid = {
        "C": [1e2, 5e2, 1e3, 5e3, 1e4, 5e4],
        "gamma": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    }
    
    # Set class_weight='balanced' to automatically adjust weights inversely proportional to class frequencies
    svc = SVC(kernel="rbf", class_weight="balanced", random_state=42)
    grid_search = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train_pca, y_train)
    
    print("\nGrid Search Complete!")
    print(f"Best hyperparameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_*100:.2f}%")
    
    return grid_search.best_estimator_


def evaluate_model(model, X_test_pca, y_test, target_names):
    """
    Evaluates the model on test data, outputs classification report metrics,
    and plots a beautifully heatmapped confusion matrix.
    """
    print("\n[Step 5] Evaluating the model on the test set...")
    y_pred = model.predict(X_test_pca)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Test Accuracy: {accuracy*100:.2f}%")
    
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)
    
    # Compute and plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", 
        xticklabels=target_names, yticklabels=target_names,
        cbar=True, square=True, annot_kws={"size": 10}
    )
    plt.title("Confusion Matrix of SVM Classifier", weight="bold", pad=20)
    plt.xlabel("Predicted Label", labelpad=10)
    plt.ylabel("True Label", labelpad=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()
    print("Saved confusion matrix heatmap to 'confusion_matrix.png'\n")
    
    return y_pred


def visualize_predictions(X_test, y_test, y_pred, target_names, h, w):
    """
    Plots a gallery of test images labeled with predicted and true identities.
    Color-codes the label: green for correct, red for incorrect.
    """
    print("[Step 6] Plotting predictions gallery...")
    fig, axes = plt.subplots(4, 6, figsize=(14, 10))
    fig.suptitle("SVM Predictions Gallery (Green = Correct, Red = Incorrect)", weight="bold", y=0.98, fontsize=15)
    
    # We plot the first 24 images in the test set
    for i, ax in enumerate(axes.flat):
        # Reshape the flat pixel vector back to its 2D image dimension
        ax.imshow(X_test[i].reshape(h, w), cmap="gray")
        
        pred_name = target_names[y_pred[i]].split()[-1]  # Get last name for shorter label
        true_name = target_names[y_test[i]].split()[-1]
        
        # Color label green if correct, red if incorrect
        color = "green" if y_pred[i] == y_test[i] else "red"
        
        ax.set_title(f"Pred: {pred_name}\nTrue: {true_name}", color=color, fontsize=9, fontweight="bold")
        ax.axis("off")
        
    plt.tight_layout()
    plt.savefig("predictions_gallery.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved prediction diagnostics gallery to 'predictions_gallery.png'\n")


def main():
    print("=========================================================================")
    print("        IMAGE CLASSIFICATION PIPELINE USING PCA AND SVM (RBF KERNEL)     ")
    print("=========================================================================\n")
    
    # 1. Load Data
    X, y, target_names, h, w = load_and_explore_data()
    
    # 2. Preprocess & Scale (Scale features is crucial before PCA)
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(X, y)
    
    # 3. Dimensionality Reduction (PCA)
    # Using 150 components as standard for LFW people
    n_components = 150
    X_train_pca, X_test_pca, pca = apply_pca(X_train_scaled, X_test_scaled, h, w, n_components=n_components)
    
    # 4. Train Model with Hyperparameter Tuning (SVM)
    best_svm = train_svm_grid_search(X_train_pca, y_train)
    
    # 5. Evaluate Model
    y_pred = evaluate_model(best_svm, X_test_pca, y_test, target_names)
    
    # 6. Visualize Predictions
    # Note: We pass raw scaled test set X_test_scaled so we can reshape the image back to 2D for visualization
    visualize_predictions(X_test_scaled, y_test, y_pred, target_names, h, w)
    
    print("=========================================================================")
    print(" Pipeline Execution Complete! All visualization outputs saved as PNGs.  ")
    print("=========================================================================")


if __name__ == "__main__":
    main()
