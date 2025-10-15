import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

def scale_features(df_grouped):
    """Scale features for clustering."""
    X = df_grouped[['stop_id', 'avg_passengers']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def find_best_k(X_scaled, k_range=(2, 11)):
    """Find the best number of clusters using silhouette score."""
    scores = []
    for k in range(*k_range):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        scores.append((k, silhouette_score(X_scaled, labels)))
    best_k = max(scores, key=lambda x: x[1])[0]
    return best_k, scores

def apply_kmeans(df_grouped, best_k):
    """Apply K-Means clustering."""
    X_scaled, _ = scale_features(df_grouped)
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    df_grouped['cluster'] = kmeans.fit_predict(X_scaled)
    return df_grouped

def apply_dbscan(df_grouped, eps=0.5, min_samples=5):
    """Apply DBSCAN clustering."""
    X_scaled, _ = scale_features(df_grouped)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df_grouped['dbscan_cluster'] = dbscan.fit_predict(X_scaled)
    return df_grouped

def evaluate_clusters(df_grouped):
    """Evaluate clusters with silhouette score and stats."""
    X_scaled, _ = scale_features(df_grouped)
    sil = silhouette_score(X_scaled, df_grouped['cluster'])
    cluster_stats = df_grouped.groupby('cluster').agg(
        count=('stop_id', 'count'),
        avg_passengers=('avg_passengers', 'mean'),
        total_passengers=('total_passengers', 'sum')
    )
    return sil, cluster_stats

if __name__ == "__main__":
    from preprocess import load_data, preprocess_data

    df = load_data("../data/public_transport.parquet")
    df_grouped = preprocess_data(df)

    X_scaled, _ = scale_features(df_grouped)
    best_k, scores = find_best_k(X_scaled)
    print(f"Best k: {best_k}, Scores: {scores}")

    df_clustered = apply_kmeans(df_grouped, best_k)
    sil, stats = evaluate_clusters(df_clustered)
    print(f"Silhouette: {sil}")
    print(stats)