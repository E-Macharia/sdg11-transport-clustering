import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from src.preprocess import load_data, preprocess_data
from src.clustering import scale_features, find_best_k, apply_kmeans, evaluate_clusters

st.title("AI-Driven Public Transport Route Optimization (SDG 11)")

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload public_transport.parquet", type="parquet")

if uploaded_file is not None:
    df = pd.read_parquet(uploaded_file)
    df_grouped = preprocess_data(df)
else:
    st.warning("Please upload a public_transport.parquet file to proceed.")
    st.stop()

st.header("Data Overview")
st.write("Aggregated data shape:", df_grouped.shape)
st.dataframe(df_grouped.head())

st.header("Clustering")
X_scaled, _ = scale_features(df_grouped)
best_k, scores = find_best_k(X_scaled)
st.write(f"Best number of clusters: {best_k}")
st.line_chart(pd.DataFrame(scores, columns=['k', 'silhouette']))

df_clustered = apply_kmeans(df_grouped, best_k)
sil, stats = evaluate_clusters(df_clustered)
st.write(f"Silhouette Score: {sil}")
st.dataframe(stats)

st.header("Visualization")
fig, ax = plt.subplots(figsize=(10, 6))
for cluster in range(best_k):
    cluster_data = df_clustered[df_clustered['cluster'] == cluster]
    ax.scatter(cluster_data['stop_id'], cluster_data['avg_passengers'], label=f'Cluster {cluster}')
ax.set_xlabel('Stop ID (Location Proxy)')
ax.set_ylabel('Average Passengers')
ax.set_title('Clusters of Transport Stops by Demand')
ax.legend()
st.pyplot(fig)

st.header("Interpretation")
st.write("Clusters with high avg_passengers indicate high-demand zones. Optimize routes by increasing frequency in these areas.")