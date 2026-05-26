#!/usr/bin/env python3
"""
Custom Image Classification Pipeline for Gender Recognition (Men vs Women).
Created for Machine Learning Assignment #4.

This script implements:
1. Data Loading: Loading local face images from 'Data/Training' (male/female folders).
2. Image Processing: Resizing to 64x64, converting to grayscale, and flattening.
3. Preprocessing: Stratified Train/Test Split and StandardScaler normalization.
4. PCA: Extracting the top Principal Components (Gender-Eigenfaces) and explained variance.
5. SVM: Hyperparameter tuning via 5-fold cross-validated GridSearchCV.
6. Evaluation: Test accuracy, classification report, confusion matrix, and prediction gallery.

Author: Antigravity AI
Date: May 2026
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
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


def load_gender_dataset(data_dir="Data/Training", img_size=(64, 64), max_samples_per_class=1000):
    """
    Loads local images of men and women from subfolders.
    Resizes them to 64x64 and converts to grayscale for consistent PCA dimensionality reduction.
    """
    print(f"[Step 1] Loading local gender dataset from '{data_dir}'...")
    X, y = [], []
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    # Filter classes to ensure we only load male/female or men/women
    class_map = {cls: idx for idx, cls in enumerate(classes)}
    print(f"Detected classes: {class_map}")
    
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Limit dataset size if requested, to keep execution fast
        if max_samples_per_class and len(files) > max_samples_per_class:
            print(f"  Class '{cls}' contains {len(files)} files. Limiting to {max_samples_per_class} for optimal training speed.")
            files = files[:max_samples_per_class]
        else:
            print(f"  Class '{cls}' contains {len(files)} images.")
            
        loaded_count = 0
        for fname in files:
            fpath = os.path.join(cls_path, fname)
            try:
                # Open image, convert to Grayscale (L), and resize
                img = Image.open(fpath).convert("L").resize(img_size)
                # Flatten the 2D pixel array to 1D feature vector and normalize to [0, 1]
                X.append(np.array(img).flatten().astype(np.float32) / 255.0)
                y.append(class_map[cls])
                loaded_count += 1
            except Exception as e:
                # Silently skip corrupt or unreadable files
                pass
        print(f"    Successfully loaded {loaded_count} images of '{cls}'.")
        
    X = np.array(X)
    y = np.array(y)
    
    print("\n--- Gender Dataset Summary ---")
    print(f"Total loaded samples: {X.shape[0]}")
    print(f"Image dimensions: {img_size[0]}x{img_size[1]} grayscale pixels = {X.shape[1]} features")
    for cls, idx in class_map.items():
        count = np.sum(y == idx)
        print(f"  - Class '{cls}' ({idx}): {count} samples ({count/len(y)*100:.1f}%)")
    print("------------------------------\n")
    
    # Plot some sample gender faces
    fig, axes = plt.subplots(3, 5, figsize=(12, 8))
    fig.suptitle("Sample Face Images from Gender Dataset", weight="bold", y=0.98)
    for i, ax in enumerate(axes.flat):
        if i >= len(y):
            ax.axis("off")
            continue
        idx = np.random.randint(0, len(y))
        ax.imshow(X[idx].reshape(img_size), cmap="gray")
        ax.set_title(classes[y[idx]], fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("gender_samples.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved sample images plot to 'gender_samples.png'")
    
    return X, y, classes, img_size[0], img_size[1]


def preprocess_gender_data(X, y):
    """
    Splits the dataset and standardizes the features.
    """
    print("[Step 2] Splitting and scaling dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_test


def apply_gender_pca(X_train, X_test, h, w, n_components=100):
    """
    Applies PCA and plots the explained variance and eigenfaces.
    """
    print(f"[Step 3] Applying PCA (Dimensionality Reduction to {n_components} components)...")
    pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True, random_state=42).fit(X_train)
    
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    print(f"Total explained variance by {n_components} components: {explained_variance[-1]*100:.2f}%")
    
    # 1. Plot Explained Variance
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_components + 1), explained_variance, marker='o', linestyle='-', color='#E28B4A', markevery=5)
    plt.axhline(y=0.90, color='r', linestyle='--', alpha=0.7, label='90% Explained Variance')
    plt.title("Gender PCA: Cumulative Explained Variance vs. Components", weight="bold")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("gender_variance.png", dpi=300)
    plt.close()
    print("Saved cumulative explained variance plot to 'gender_variance.png'")
    
    # 2. Visualize Gender-Eigenfaces
    eigenfaces = pca.components_.reshape((n_components, h, w))
    fig, axes = plt.subplots(3, 5, figsize=(12, 8))
    fig.suptitle("Gender-Eigenfaces: First 15 Components", weight="bold", y=0.98)
    for i, ax in enumerate(axes.flat):
        ax.imshow(eigenfaces[i], cmap="magma")
        ax.set_title(f"PC {i+1}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("gender_eigenfaces.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved first 15 Gender-Eigenfaces plot to 'gender_eigenfaces.png'\n")
    
    return X_train_pca, X_test_pca, pca


def train_gender_svm(X_train_pca, y_train):
    """
    Trains an SVM using GridSearchCV to find optimal hyperparameters.
    """
    print("[Step 4] Training SVM classifier and optimizing hyperparameters...")
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": [0.0001, 0.001, 0.01, 0.1]
    }
    
    svc = SVC(kernel="rbf", class_weight="balanced", random_state=42)
    grid_search = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train_pca, y_train)
    
    print("\nGrid Search Complete!")
    print(f"Best hyperparameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_*100:.2f}%")
    
    return grid_search.best_estimator_


def evaluate_gender_model(model, X_test_pca, y_test, classes):
    """
    Evaluates gender classifier performance.
    """
    print("\n[Step 5] Evaluating the model on the test set...")
    y_pred = model.predict(X_test_pca)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Gender Classification Accuracy: {accuracy*100:.2f}%")
    
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=classes)
    print(report)
    
    # Compute and plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples", 
        xticklabels=classes, yticklabels=classes,
        cbar=True, square=True
    )
    plt.title("Confusion Matrix: Gender SVM Classifier", weight="bold", pad=20)
    plt.xlabel("Predicted Gender", labelpad=10)
    plt.ylabel("True Gender", labelpad=10)
    plt.tight_layout()
    plt.savefig("gender_confusion_matrix.png", dpi=300)
    plt.close()
    print("Saved confusion matrix heatmap to 'gender_confusion_matrix.png'\n")
    
    return y_pred


def visualize_gender_predictions(X_test_raw, y_test, y_pred, classes, h, w):
    """
    Plots a diagnostic prediction gallery.
    """
    print("[Step 6] Plotting predictions gallery...")
    fig, axes = plt.subplots(4, 6, figsize=(14, 10))
    fig.suptitle("Gender Classification Predictions (Green = Correct, Red = Incorrect)", weight="bold", y=0.98, fontsize=15)
    
    for i, ax in enumerate(axes.flat):
        if i >= len(y_test):
            ax.axis("off")
            continue
        # Display the actual raw test image (grayscale)
        ax.imshow(X_test_raw[i].reshape(h, w), cmap="gray")
        
        pred_label = classes[y_pred[i]]
        true_label = classes[y_test[i]]
        
        color = "green" if y_pred[i] == y_test[i] else "red"
        
        ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}", color=color, fontsize=10, fontweight="bold")
        ax.axis("off")
        
    plt.tight_layout()
    plt.savefig("gender_predictions.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved prediction diagnostics gallery to 'gender_predictions.png'\n")


def main():
    print("=========================================================================")
    print("      CUSTOM GENDER CLASSIFICATION PIPELINE: MEN VS WOMEN (PCA + SVM)    ")
    print("=========================================================================\n")
    
    # 1. Load Custom Dataset
    # We set a limit of 1000 images per class for fast execution, but it covers the dataset perfectly
    X, y, classes, h, w = load_gender_dataset(data_dir="Data/Training", img_size=(64, 64), max_samples_per_class=1000)
    
    # 2. Split and Scale
    X_train_scaled, X_test_scaled, y_train, y_test, X_test_raw = preprocess_gender_data(X, y)
    
    # 3. PCA
    # Using 80 components which is highly optimal for 64x64 gender face images
    n_components = 80
    X_train_pca, X_test_pca, pca = apply_gender_pca(X_train_scaled, X_test_scaled, h, w, n_components=n_components)
    
    # 4. Train with Cross-Validated Hyperparameter Tuning
    best_svm = train_gender_svm(X_train_pca, y_train)
    
    # 5. Evaluate Model
    y_pred = evaluate_gender_model(best_svm, X_test_pca, y_test, classes)
    
    # 6. Prediction Diagnostics
    visualize_gender_predictions(X_test_raw, y_test, y_pred, classes, h, w)
    
    print("=========================================================================")
    print(" Gender Pipeline Execution Complete! All visualization outputs saved.   ")
    print("=========================================================================")


if __name__ == "__main__":
    main()
