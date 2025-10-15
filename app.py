import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# ✅ Dynamically add project root to Python path (for Streamlit Cloud)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir))
sys.path.append(project_root)

# ✅ Import functions from your src module
from src.preprocess import load_data, preprocess_data
from src.clustering import scale_features, find_best_k, apply_kmeans, evaluate_clusters

# ------------------ Streamlit UI ------------------

st.title("🚍 AI-Driven Public Transport Route Optimization (SDG 11)")
st.markdown("### Using Unsupervised Machine Learning (K-Means) to Identify High-Demand Transport Zones")

# Sidebar for file upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload `public_transport.parquet`", type="parquet")

# ------------------ Data Loading ------------------

if uploaded_file is not None:
    try:
        df = pd.read_parquet(uploaded_file)
        df_grouped = preprocess_data(df)
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}. Please ensure it's a valid parquet file.")
        st.stop()
else:
    st.warning("⚠️ Please upload a `public_transport.parquet` file to proceed.")
    st.stop()

# ------------------ Data Overview ------------------

st.header("📊 Data Overview")
st.write("**Aggregated Data Shape:**", df_grouped.shape)
st.dataframe(df_grouped.head())

# ------------------ Clustering Section ------------------

st.header("🔢 Clustering Analysis")
X_scaled, _ = scale_features(df_grouped)

# Find optimal number of clusters
best_k, scores = find_best_k(X_scaled)
st.write(f"**Optimal number of clusters (k):** {best_k}")

# Plot silhouette scores for each k
st.line_chart(pd.DataFrame(scores, columns=['k', 'silhouette']))

# Apply clustering with best k
df_clustered = apply_kmeans(df_grouped, best_k)

# Evaluate results
sil, stats = evaluate_clusters(df_clustered)
st.write(f"**Silhouette Score:** {sil:.4f}")
st.dataframe(stats)

# ------------------ Visualization ------------------

st.header("🗺️ Visualization of Clusters")
fig, ax = plt.subplots(figsize=(10, 6))
for cluster in range(best_k):
    cluster_data = df_clustered[df_clustered['cluster'] == cluster]
    ax.scatter(
        cluster_data['stop_id'],
        cluster_data['avg_passengers'],
        label=f'Cluster {cluster}',
        s=80
    )

ax.set_xlabel('Stop ID (as a proxy for location)')
ax.set_ylabel('Average Passengers')
ax.set_title('Clusters of Transport Stops by Demand')
ax.legend()
st.pyplot(fig)

# ------------------ Interpretation ------------------

st.header("🧠 Interpretation & Insights")
st.success(
    "Clusters with high average passenger counts represent **high-demand zones**. "
    "City planners can optimize routes by **increasing frequency** or **adding new stops** "
    "in these regions to improve transport efficiency and support **SDG 11 – Sustainable Cities and Communities**."
)
