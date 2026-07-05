import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, RFE
import umap
import shap

from amc.visualization.base import setup_publication_style

def main():
    features_path = os.path.join("dataset", "features.csv")
    output_dir = os.path.join("results", "analysis")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(features_path):
        print(f"Error: features.csv not found at {features_path}. Please run extract_features.py first.")
        return

    print("Loading extracted features...")
    df = pd.read_csv(features_path)

    # Exclude metadata columns to isolate features
    meta_cols = ["file_path", "modulation", "snr_db"]
    feature_cols = [col for col in df.columns if col not in meta_cols]

    X_raw = df[feature_cols].values
    y_str = df["modulation"].values
    snr = df["snr_db"].values

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)
    classes = label_encoder.classes_

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    setup_publication_style()

    print("1. Plotting Feature vs SNR...")
    # Select key features to show over SNR
    key_features = ["papr", "inst_phase_std", "C42", "spectral_flatness"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, feat in enumerate(key_features):
        if feat in df.columns:
            sns.lineplot(data=df, x="snr_db", y=feat, hue="modulation", ax=axes[idx], marker="o", errorbar="ci")
            axes[idx].set_title(f"{feat} vs SNR")
            axes[idx].set_xlabel("SNR (dB)")
            axes[idx].set_ylabel("Value")
            axes[idx].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "feature_vs_snr.png"))
    plt.close(fig)

    print("2. Plotting Histograms, Boxplots, and Violins for key feature (papr)...")
    if "papr" in df.columns:
        # Histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=df, x="papr", hue="modulation", kde=True, element="step", ax=ax, alpha=0.4)
        ax.set_title("PAPR Distribution by Modulation")
        fig.savefig(os.path.join(output_dir, "papr_histogram.png"))
        plt.close(fig)

        # Boxplot
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x="modulation", y="papr", ax=ax, palette="Set2")
        ax.set_title("PAPR Boxplot by Modulation")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.savefig(os.path.join(output_dir, "papr_boxplot.png"))
        plt.close(fig)

        # Violin plot
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.violinplot(data=df, x="modulation", y="papr", ax=ax, palette="muted", inner="quartile")
        ax.set_title("PAPR Violin Plot by Modulation")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.savefig(os.path.join(output_dir, "papr_violin.png"))
        plt.close(fig)

    print("3. Generating Correlation Matrix...")
    # Calculate correlation matrix
    corr = df[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", xticklabels=True, yticklabels=True, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    plt.close(fig)

    # Subsample data for expensive reductions (PCA/TSNE/UMAP)
    # Using 1200 samples (balanced across classes) for cleaner plots and fast processing
    df_sample = df.groupby("modulation").sample(n=min(150, len(df)//len(classes)), random_state=42).reset_index(drop=True)
    X_sub = scaler.transform(df_sample[feature_cols].values)
    y_sub_str = df_sample["modulation"].values


    print("4. Executing PCA Dimensionality Reduction...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sub)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_sub_str, ax=ax, palette="tab10", alpha=0.8)
    ax.set_title(f"PCA Dimension Reduction (Explained Var: {sum(pca.explained_variance_ratio_)*100:.1f}%)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    fig.savefig(os.path.join(output_dir, "dim_reduction_pca.png"))
    plt.close(fig)

    print("5. Executing t-SNE Dimensionality Reduction...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_sub_str, ax=ax, palette="tab10", alpha=0.8)
    ax.set_title("t-SNE Dimension Reduction")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    fig.savefig(os.path.join(output_dir, "dim_reduction_tsne.png"))
    plt.close(fig)

    print("6. Executing UMAP Dimensionality Reduction...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_sub)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y_sub_str, ax=ax, palette="tab10", alpha=0.8)
    ax.set_title("UMAP Dimension Reduction")
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    fig.savefig(os.path.join(output_dir, "dim_reduction_umap.png"))
    plt.close(fig)

    print("7. Training Random Forest & Extracting Feature Importances...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_importances = rf.feature_importances_

    print("8. Computing Mutual Information Scores...")
    mi_scores = mutual_info_classif(X, y, random_state=42)

    print("9. Running Recursive Feature Elimination (RFE)...")
    # Using a smaller RF estimator for speed in RFE
    rfe_estimator = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=rfe_estimator, n_features_to_select=1, step=2)
    rfe.fit(X, y)
    rfe_rankings = rfe.ranking_

    print("10. Conducting SHAP Analysis...")
    # Generate SHAP values using TreeExplainer
    # Use a subsample to speed up explainer background dataset mapping
    X_shap_bg = shap.utils.sample(X, 200)
    explainer = shap.TreeExplainer(rf, data=X_shap_bg)
    # Calculate shap values for test instances
    X_shap_test = shap.utils.sample(X, 100)
    # SHAP returns a list of arrays for multi-class; we average their absolute values or select a single class.
    # We will compute SHAP values for the first class for simplicity or average absolute contribution.
    shap_values = explainer.shap_values(X_shap_test)
    
    # Generate SHAP summary plot (save via matplotlib redirection)
    plt.figure(figsize=(10, 6))
    # SHAP summary plot can plot multi-class lists directly in newer versions
    shap.summary_plot(shap_values, X_shap_test, feature_names=feature_cols, show=False)
    plt.title("SHAP Feature Summary (Absolute Contribution)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"))
    plt.close()

    print("11. Ranking Features...")
    # Compile Rankings DataFrame
    rank_df = pd.DataFrame({
        "feature": feature_cols,
        "rf_importance": rf_importances,
        "mutual_info": mi_scores,
        "rfe_rank": rfe_rankings
    })

    # Normalized score combines importance and mutual info, penalizing high RFE rank
    # Final Rank Score = (rf_importance / max_rf) + (mutual_info / max_mi) - (rfe_rank / max_rfe)
    rank_df["rf_norm"] = rank_df["rf_importance"] / rank_df["rf_importance"].max()
    rank_df["mi_norm"] = rank_df["mutual_info"] / rank_df["mutual_info"].max()
    rank_df["rfe_norm"] = 1.0 - (rank_df["rfe_rank"] / rank_df["rfe_rank"].max()) # High is better
    
    rank_df["combined_score"] = (rank_df["rf_norm"] + rank_df["mi_norm"] + rank_df["rfe_norm"]) / 3.0
    rank_df = rank_df.sort_values(by="combined_score", ascending=False).reset_index(drop=True)
    
    # Save rankings
    rankings_csv_path = os.path.join(output_dir, "feature_rankings.csv")
    rank_df.to_csv(rankings_csv_path, index=False)

    # Plot top 15 features
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=rank_df.head(15), x="combined_score", y="feature", ax=ax, palette="viridis")
    ax.set_title("Top 15 Ranked Features")
    ax.set_xlabel("Combined Quality Score")
    fig.savefig(os.path.join(output_dir, "top_ranked_features.png"))
    plt.close(fig)

    print(f"\nFeature analysis completed successfully.")
    print(f"Figures and rankings table saved in: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()
