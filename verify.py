import os
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score


def load_and_preprocess_data(filepath='../data/mnist_train.csv', n_samples=10000, random_state=42):
    """
    Loads the MNIST dataset and preprocesses it.
    
    Args:
        filepath (str): Path to the mnist CSV file.
        n_samples (int): Number of samples to use (for performance reasons with DBSCAN).
                         Set to None to use all data.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: (scaled_features, true_labels)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")
        
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # We assume the first column is 'label' and the rest are pixels
    if 'label' in df.columns:
        labels = df['label']
        features = df.drop('label', axis=1)
    else:
        # Assuming first column is label if not explicitly named
        labels = df.iloc[:, 0]
        features = df.iloc[:, 1:]
        
    if n_samples is not None and n_samples < len(df):
        print(f"Sampling {n_samples} random records out of {len(df)}...")
        df_sample = df.sample(n=n_samples, random_state=random_state)
        if 'label' in df_sample.columns:
            labels = df_sample['label']
            features = df_sample.drop('label', axis=1)
        else:
            labels = df_sample.iloc[:, 0]
            features = df_sample.iloc[:, 1:]
            
    print("Normalizing features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, labels.values




def apply_pca(features, n_components=2):
    """
    Applies PCA to reduce dimensionality of the features.
    
    Args:
        features (numpy.ndarray): The scaled feature matrix.
        n_components (int or float): Number of components to keep, 
                                     or fraction of variance to explain.
                                     
    Returns:
        tuple: (pca_features, pca_model)
    """
    print(f"Applying PCA (n_components={n_components})...")
    pca = PCA(n_components=n_components, random_state=42)
    pca_features = pca.fit_transform(features)
    
    explained_variance = sum(pca.explained_variance_ratio_)
    print(f"Total explained variance by components: {explained_variance:.4f}")
    
    return pca_features, pca




def run_dbscan(features, eps=0.5, min_samples=5):
    """
    Applies DBSCAN clustering to the feature matrix.
    
    Args:
        features (numpy.ndarray): The feature matrix (could be PCA reduced).
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        
    Returns:
        tuple: (cluster_labels, dbscan_model)
    """
    print(f"Running DBSCAN (eps={eps}, min_samples={min_samples})...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Estimated number of clusters: {n_clusters}")
    print(f"Estimated number of noise points: {n_noise}")
    
    return labels, dbscan




def run_kmeans(features, n_clusters=10, random_state=42):
    """
    Applies KMeans clustering to the feature matrix.
    
    Args:
        features (numpy.ndarray): The feature matrix (could be PCA reduced).
        n_clusters (int): Number of clusters.
        random_state (int): Random seed.
        
    Returns:
        tuple: (cluster_labels, kmeans_model)
    """
    print(f"Running KMeans (n_clusters={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    labels = kmeans.fit_predict(features)
    
    return labels, kmeans




def evaluate_clusters(features, cluster_labels, true_labels=None):
    """
    Evaluates clustering performance.
    
    Args:
        features: Feature matrix used for clustering.
        cluster_labels: Predicted labels.
        true_labels: True labels (if available).
        
    Returns:
        dict: Dictionary of metrics.
    """
    metrics = {}
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    metrics['n_clusters'] = n_clusters
    metrics['n_noise'] = n_noise
    
    if n_clusters > 1:
        metrics['silhouette'] = silhouette_score(features, cluster_labels)
    else:
        metrics['silhouette'] = None
        
    if true_labels is not None:
        metrics['ari'] = adjusted_rand_score(true_labels, cluster_labels)
        metrics['nmi'] = normalized_mutual_info_score(true_labels, cluster_labels)
        
    return metrics




def plot_pca_clusters(pca_features, cluster_labels, title="DBSCAN Clustering on PCA-Reduced Data", output_path=None):
    """
    Plots a 2D scatter plot of the PCA features, colored by cluster labels.
    
    Args:
        pca_features (numpy.ndarray): PCA-reduced feature matrix (must have at least 2 dims).
        cluster_labels (numpy.ndarray): Cluster labels from DBSCAN.
        output_path (str, optional): Path to save the plot.
    """
    plt.figure(figsize=(10, 8))
    
    # Identify unique labels
    unique_labels = set(cluster_labels)
    
    # Create a color palette
    palette = sns.color_palette("husl", len(unique_labels))
    
    # Plot noise points (-1) in black
    if -1 in unique_labels:
        mask_noise = (cluster_labels == -1)
        plt.scatter(pca_features[mask_noise, 0], pca_features[mask_noise, 1], 
                    c='black', label='Noise', alpha=0.3, s=10)
    
    # Plot clusters
    for i, label in enumerate(unique_labels):
        if label != -1:
            mask = (cluster_labels == label)
            plt.scatter(pca_features[mask, 0], pca_features[mask, 1], 
                        c=[palette[i]], label=f'Cluster {label}', alpha=0.6, s=15)
            
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    
    # Only show legend if we don't have too many clusters
    if len(unique_labels) <= 20:
        plt.legend()
        
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved cluster plot to {output_path}")
    
    plt.show()

def plot_true_labels(pca_features, true_labels, output_path=None):
    """
    Plots a 2D scatter plot colored by TRUE labels for comparison.
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], 
                    hue=true_labels, palette="tab10", legend="full", alpha=0.6, s=15)
    
    plt.title("PCA-Reduced Data Colored by True Labels")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved PCA true labels plot to {output_path}")
        
    plt.show()

def plot_pca_variance(pca_model, output_path=None):
    """Plots the explained variance ratio of PCA components."""
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca_model.explained_variance_ratio_), marker='o', linestyle='--')
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved PCA variance plot to {output_path}")
        
    plt.show()

def plot_cluster_comparison(pca_features, labels1, name1, labels2, name2, output_path=None):
    """Plots a side-by-side comparison of two clusterings."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # First plot
    unique1 = set(labels1)
    pal1 = sns.color_palette("husl", len(unique1))
    if -1 in unique1:
        mask = (labels1 == -1)
        ax1.scatter(pca_features[mask, 0], pca_features[mask, 1], c='black', label='Noise', alpha=0.3, s=10)
    for i, label in enumerate(unique1):
        if label != -1:
            mask = (labels1 == label)
            ax1.scatter(pca_features[mask, 0], pca_features[mask, 1], c=[pal1[i]], alpha=0.6, s=15)
    ax1.set_title(name1)
    ax1.set_xlabel("Principal Component 1")
    ax1.set_ylabel("Principal Component 2")
    
    # Second plot
    unique2 = set(labels2)
    pal2 = sns.color_palette("husl", len(unique2))
    if -1 in unique2:
        mask = (labels2 == -1)
        ax2.scatter(pca_features[mask, 0], pca_features[mask, 1], c='black', label='Noise', alpha=0.3, s=10)
    for i, label in enumerate(unique2):
        if label != -1:
            mask = (labels2 == label)
            ax2.scatter(pca_features[mask, 0], pca_features[mask, 1], c=[pal2[i]], alpha=0.6, s=15)
    ax2.set_title(name2)
    ax2.set_xlabel("Principal Component 1")
    ax2.set_ylabel("Principal Component 2")
    
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved cluster comparison plot to {output_path}")
    plt.show()




def main():
    print("Starting verification run...")
    
    # 1. Load data
    scaled_features, true_labels = load_and_preprocess_data(filepath='data/mnist_train.csv', n_samples=5000)
    
    # 2. PCA
    pca_50_features, pca_50_model = apply_pca(scaled_features, n_components=50)
    pca_2_features, pca_2_model = apply_pca(scaled_features, n_components=2)
    
    os.makedirs('outputs', exist_ok=True)
    
    # Plot PCA variance
    plot_pca_variance(pca_50_model, output_path='outputs/pca_variance.png')
    
    # Plot true labels
    plot_true_labels(pca_2_features, true_labels, output_path='outputs/true_labels.png')
    
    # 3. Clustering
    # DBSCAN
    dbscan_labels, dbscan_model = run_dbscan(pca_50_features, eps=15.0, min_samples=10)
    plot_pca_clusters(pca_2_features, dbscan_labels, output_path='outputs/dbscan_clusters.png')
    
    # KMeans
    kmeans_labels, kmeans_model = run_kmeans(pca_50_features, n_clusters=10)
    plot_pca_clusters(pca_2_features, kmeans_labels, title="KMeans Clustering on PCA-Reduced Data", output_path='outputs/kmeans_clusters.png')
    
    # Compare
    plot_cluster_comparison(pca_2_features, dbscan_labels, "DBSCAN", kmeans_labels, "KMeans", output_path='outputs/cluster_comparison.png')
    
    # 4. Metrics
    dbscan_metrics = evaluate_clusters(pca_50_features, dbscan_labels, true_labels)
    kmeans_metrics = evaluate_clusters(pca_50_features, kmeans_labels, true_labels)
    
    with open('outputs/metrics.txt', 'w') as f:
        f.write("DBSCAN Metrics:\n")
        f.write(f"  Number of clusters: {dbscan_metrics['n_clusters']}\n")
        f.write(f"  Number of noise points: {dbscan_metrics['n_noise']}\n")
        f.write(f"  Silhouette Score: {dbscan_metrics['silhouette']}\n")
        f.write(f"  ARI: {dbscan_metrics.get('ari', 'N/A')}\n")
        f.write(f"  NMI: {dbscan_metrics.get('nmi', 'N/A')}\n\n")
        
        f.write("KMeans Metrics:\n")
        f.write(f"  Number of clusters: {kmeans_metrics['n_clusters']}\n")
        f.write(f"  Silhouette Score: {kmeans_metrics['silhouette']}\n")
        f.write(f"  ARI: {kmeans_metrics.get('ari', 'N/A')}\n")
        f.write(f"  NMI: {kmeans_metrics.get('nmi', 'N/A')}\n")
        
    print("Verification completed successfully!")

if __name__ == '__main__':
    main()
