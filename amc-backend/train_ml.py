import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

# Classifiers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from amc.visualization.base import setup_publication_style
from amc.utils.visualization import plot_confusion_matrix

def train_and_evaluate_models():
    features_path = os.path.join("dataset", "features.csv")
    models_dir = os.path.join("results", "models")
    eval_dir = os.path.join("results", "ml_eval")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    if not os.path.exists(features_path):
        print(f"Error: features.csv not found at {features_path}. Please run extract_features.py first.")
        return

    print("Loading features...")
    df = pd.read_csv(features_path)

    # Exclude metadata columns
    meta_cols = ["file_path", "modulation", "snr_db"]
    feature_cols = [col for col in df.columns if col not in meta_cols]

    X_raw = df[feature_cols].values
    y_str = df["modulation"].values

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)
    classes = label_encoder.classes_

    # Split dataset into train and test sets (stratified, 80/20 split)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Save the fitted scaler
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(models_dir, "label_encoder.pkl"))

    # Instantiate all 7 classifiers
    # Keep max_iter / verbosity low to speed up and keep stdout clean
    classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM": SVC(probability=True, kernel="rbf", random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=50, eval_metric="mlogloss", random_state=42, n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=50, random_state=42, n_jobs=-1, verbosity=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
    }

    performance_records = []
    setup_publication_style()

    # Stratified K-fold for cross-validation
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize plotting of multi-model ROC Curves
    plt.figure(figsize=(10, 8))

    for name, clf in classifiers.items():
        print(f"\n--- Training {name} Classifier ---")
        
        # Fit classifier
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        # Evaluate test metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        print(f"Accuracy : {acc:.4f}")
        print(f"F1 Score : {f1:.4f}")

        # 5-fold cross validation on training set for clean validation splits
        print(f"Running 5-fold Cross Validation for {name}...")
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv_strategy, scoring="accuracy", n_jobs=-1)
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        print(f"CV Mean Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")

        # Save model binary
        model_path = os.path.join(models_dir, f"{name.lower()}_model.pkl")
        joblib.dump(clf, model_path)

        # 1. Confusion Matrix plotting
        cm = confusion_matrix(y_test, y_pred)
        fig = plot_confusion_matrix(cm, classes=list(classes), title=f"{name} Confusion Matrix")
        fig.savefig(os.path.join(eval_dir, f"{name.lower()}_confusion_matrix.png"))
        plt.close(fig)

        # 2. Multi-class ROC mapping (One-vs-Rest ROC curve average)
        # Compute macro-average ROC curve
        fpr_dict = dict()
        tpr_dict = dict()
        roc_auc_dict = dict()
        
        # Calculate OvR ROC curve for each class
        n_classes = len(classes)
        for i in range(n_classes):
            # Binary label for class i
            y_test_bin = (y_test == i).astype(int)
            y_prob_class = y_prob[:, i]
            fpr_dict[i], tpr_dict[i], _ = roc_curve(y_test_bin, y_prob_class)
            roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        mean_tpr /= n_classes

        macro_auc = auc(all_fpr, mean_tpr)

        # Add to overall multi-model ROC plot
        plt.plot(all_fpr, mean_tpr, label=f"{name} (Macro AUC = {macro_auc:.3f})", linewidth=1.5)

        # Store statistics
        performance_records.append({
            "Classifier": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "CV_Mean": cv_mean,
            "CV_Std": cv_std,
            "Macro_AUC": macro_auc
        })

    # Finalize and save macro-average ROC curves comparison plot
    plt.plot([0, 1], [0, 1], 'k--', color='#666666', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Macro-Average ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "roc_comparison.png"))
    plt.close()

    # Save summary Performance DataFrame to CSV
    df_perf = pd.DataFrame(performance_records)
    summary_path = os.path.join(eval_dir, "model_performance_summary.csv")
    df_perf.to_csv(summary_path, index=False)

    print(f"\n--- Machine Learning Pipeline completed successfully ---")
    print(f"Performance summary saved to: {os.path.abspath(summary_path)}")
    print(f"Models saved under: {os.path.abspath(models_dir)}")
    print(f"Evaluation plots saved under: {os.path.abspath(eval_dir)}")

if __name__ == "__main__":
    train_and_evaluate_models()
