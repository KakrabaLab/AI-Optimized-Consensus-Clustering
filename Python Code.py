# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 17:49:04 2025
# Code and Computational Workflow for AI-Optimized Amino Acid Consensus Clustering Analysis###
## This code usese the file "amino_acid3.csv" from our github page. You can adapt it to other files of your choice.


@authors: Samuel Kakraba, Aayire C. Yadem, Kuukua E. Abraham
@address:Department of Biostatistics and Data Science, Tulane University Celia Scott Weatherhead School of Public Health and Tropical Medicine, Tulane University, USA
@contact:skakraba@tulane.edu/kakrabasamuel@gmail.com
@citation:Kakraba S., Yadem C.A., Abraham E. K.,"Code and Computational Workflow for AI-Optimized Amino Acid Consensus Clustering Analysis"
"""

import sys
import subprocess
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, 
                            davies_bouldin_score)
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Install required packages automatically
required_packages = [
    'gap-statistics',
    'scikit-learn',
    'scipy',
    'matplotlib',
    'pandas',
    'numpy',
    'joblib'
]

subprocess.call([sys.executable, '-m', 'pip', 'install'] + required_packages)

from gap_statistic import OptimalK

def load_data(filename):
    """Load and preprocess molecular descriptor data"""
    df = pd.read_csv(filename)
    ## requires the file "amino_acid3.csv" from our github page
    # Transpose to get amino acids as rows
    transposed = df.T.reset_index()
    transposed.columns = transposed.iloc[0]
    transposed = transposed.drop(0)
    
    # Rename columns and set index
    transposed = transposed.rename(columns={'index': 'Amino_Acid'})
    transposed = transposed.reset_index(drop=True)
    
    # Convert all feature columns to numeric
    for col in transposed.columns[1:]:
        transposed[col] = pd.to_numeric(transposed[col])
    
    return transposed['Amino_Acid'], transposed.drop('Amino_Acid', axis=1)

def normalize_data(X):
    """Z-score normalization of molecular descriptors"""
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def hierarchical_clustering(X, linkage_method='ward', metric='manhattan'):
    """Perform hierarchical clustering with validation metrics"""
    if linkage_method == 'ward':
        Z = linkage(X, method=linkage_method, metric='euclidean')
    else:
        metric = 'cityblock' if metric == 'manhattan' else metric
        Z = linkage(X, method=linkage_method, metric=metric)
    coph_corr, _ = cophenet(Z, pdist(X, metric=(metric if linkage_method != 'ward' else 'euclidean')))
    return Z, coph_corr

def evaluate_clusters(X, linkage_method='average', metric='manhattan', max_clusters=5):
    """Calculate clustering validation metrics"""
    metric = 'cityblock' if metric == 'manhattan' else metric
    Z = linkage(X, method=linkage_method, metric=metric)
    
    metrics = {
        'Silhouette': [],
        'Calinski-Harabasz': [],
        'Davies-Bouldin': [],
        'Inertia': []
    }
    
    for n in range(2, max_clusters+1):
        labels = AgglomerativeClustering(
            n_clusters=n, 
            linkage=linkage_method,
            affinity=metric
        ).fit_predict(X)
        
        metrics['Silhouette'].append(silhouette_score(X, labels))
        metrics['Calinski-Harabasz'].append(calinski_harabasz_score(X, labels))
        metrics['Davies-Bouldin'].append(davies_bouldin_score(X, labels))
        metrics['Inertia'].append(calculate_inertia(X, labels))
    
    return pd.DataFrame(metrics, index=range(2, max_clusters+1))

def calculate_inertia(X, labels):
    """Compute within-cluster sum of squares"""
    centroids = [X[labels == i].mean(axis=0) for i in np.unique(labels)]
    return sum(np.linalg.norm(X[labels == i] - centroids[i], axis=1).sum() 
               for i in np.unique(labels))

def gap_statistic(X, max_clusters=5):
    """Calculate optimal clusters using gap statistic"""
    optimalK = OptimalK(parallel_backend='joblib')
    n_clusters = optimalK(X, cluster_array=np.arange(1, max_clusters+1))
    return n_clusters

def consensus_clustering(X, n_clusters, linkage_method='average', metric='cityblock', 
                         n_iter=100, subsample_ratio=0.8):
    """Consensus clustering with configurable parameters"""
    n_samples = X.shape[0]
    subsample_size = int(n_samples * subsample_ratio)
    consensus_matrix = np.zeros((n_samples, n_samples))

    def _subsample_clustering():
        idx = np.random.choice(n_samples, subsample_size, replace=False)
        sub_X = X[idx]
        labels = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method,
            affinity=metric
        ).fit_predict(sub_X)
        return idx, labels

    results = Parallel(n_jobs=-1)(delayed(_subsample_clustering)() for _ in range(n_iter))

    for idx, labels in results:
        for i in range(len(idx)):
            for j in range(i+1, len(idx)):
                if labels[i] == labels[j]:
                    consensus_matrix[idx[i], idx[j]] += 1
                    consensus_matrix[idx[j], idx[i]] += 1

    consensus_matrix /= n_iter
    return consensus_matrix

def compare_algorithms(X, n_clusters):
    """Compare different clustering algorithms"""
    algorithms = {
        'Agglomerative': AgglomerativeClustering(n_clusters=n_clusters),
        'KMeans': KMeans(n_clusters=n_clusters),
        'DBSCAN': DBSCAN()
    }
    
    results = {}
    for name, model in algorithms.items():
        labels = model.fit_predict(X)
        if len(np.unique(labels)) > 1:
            results[name] = {
                'Silhouette': silhouette_score(X, labels),
                'Calinski-Harabasz': calinski_harabasz_score(X, labels),
                'Davies-Bouldin': davies_bouldin_score(X, labels)
            }
    return pd.DataFrame(results).T

def plot_dendrogram(Z, labels):
    """Custom dendrogram plotting"""
    plt.figure(figsize=(12, 7))
    dendrogram(
        Z, 
        labels=labels, 
        orientation='right',
        leaf_font_size=12, 
        color_threshold=0.7 * max(Z[:, 2])
    )
    plt.xlabel('Distance')
    plt.ylabel('Amino Acids')
    plt.title('Hierarchical Clustering Dendrogram')
    plt.tight_layout()
    plt.savefig('dendrogram.png', dpi=300)
    plt.close()

def main(input_csv):
    # 1. Data Loading and Preprocessing
    aa_names, X = load_data(input_csv)
    X_normalized = normalize_data(X)
    
    # 2. Linkage Method Evaluation & Selection
    linkage_methods = ['ward', 'average', 'complete', 'single']
    linkage_results = []
    
    for method in linkage_methods:
        try:
            metric = 'euclidean' if method == 'ward' else 'cityblock'
            Z, coph_corr = hierarchical_clustering(
                X_normalized, 
                linkage_method=method, 
                metric=metric
            )
            linkage_results.append({
                'Method': method,
                'Cophenetic_Correlation': coph_corr
            })
        except Exception as e:
            linkage_results.append({
                'Method': method,
                'Cophenetic_Correlation': np.nan,
                'Error': str(e)
            })
    
    # Select best linkage method
    linkage_df = pd.DataFrame(linkage_results)
    best_method = linkage_df.loc[linkage_df['Cophenetic_Correlation'].idxmax(), 'Method']
    
    # 3. Optimal Cluster Determination
    optimal_metric = 'euclidean' if best_method == 'ward' else 'cityblock'
    metrics_df = evaluate_clusters(
        X_normalized, 
        linkage_method=best_method, 
        metric=optimal_metric
    )
    
    # Additional validation with Gap Statistic
    gap_clusters = gap_statistic(X_normalized, max_clusters=5)
    
    # Find optimal clusters using multiple criteria
    optimal_clusters = metrics_df['Silhouette'].idxmax()
    
    # 4. Consensus Clustering with Optimal Parameters
    consensus_mat = consensus_clustering(
        X_normalized, 
        n_clusters=optimal_clusters,
        linkage_method=best_method,
        metric=optimal_metric
    )
    
    # 5. Algorithm Comparison
    algorithm_comparison = compare_algorithms(X_normalized, optimal_clusters)
    
    # 6. Visualization with Optimal Parameters
    Z_optimal, _ = hierarchical_clustering(
        X_normalized, 
        linkage_method=best_method, 
        metric=optimal_metric
    )
    plot_dendrogram(Z_optimal, aa_names.values)
    
    # 7. Save Results
    linkage_df.to_csv('linkage_metrics.csv', index=False)
    metrics_df.to_csv('cluster_metrics.csv')
    np.savetxt('consensus_matrix.csv', consensus_mat, delimiter=',')
    algorithm_comparison.to_csv('algorithm_comparison.csv')
    
    print(f"Optimal parameters: {best_method} linkage, {optimal_clusters} clusters")
    print("Results saved to CSV files and dendrogram.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Amino Acid Clustering Pipeline')
    parser.add_argument('input_csv', help='Path to molecular descriptors CSV file')
    args = parser.parse_args()
    main(args.input_csv)
