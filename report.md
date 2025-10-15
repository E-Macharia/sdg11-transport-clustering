# AI-Driven Optimization of Public Transport Routes for Sustainable Cities (SDG 11)

## Summary

This project applies unsupervised machine learning (K-Means clustering) to public transport passenger data to identify high-demand zones and optimize routes, supporting SDG 11 by improving urban mobility and reducing emissions.

## Tools Used

- Python
- pandas, scikit-learn, matplotlib, streamlit
- Dataset: Public Transport Passenger Counts (Kaggle)

## Methodology

1. **Data Preprocessing**: Cleaned and aggregated passenger data by stop_id.
2. **Clustering**: Used K-Means with optimal k selected via silhouette score.
3. **Visualization**: Scatter plots of clusters.
4. **Evaluation**: Silhouette score and cluster statistics.

## Results

- Best k: [To be filled after running]
- Silhouette Score: [To be filled]
- High-demand clusters identified for route optimization.

## Interpretation

Clusters with higher average passengers represent busy zones. Recommendations: Increase bus frequency in these areas to reduce wait times and congestion.

## Limitations

- No geospatial coordinates; used stop_id as proxy.
- Assumes stops are geographically ordered.

## Future Work

- Integrate real lat/lon data.
- Use reinforcement learning for dynamic routing.