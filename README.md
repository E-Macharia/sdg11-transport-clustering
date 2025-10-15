# AI-Driven Optimization of Public Transport Routes for Sustainable Cities (SDG 11)

This project uses unsupervised learning (clustering) to identify high-demand transport zones and optimize public transport routes, contributing to SDG 11: Sustainable Cities and Communities.

## Project Structure

```
WEEK_2_ASSIGNMENT/
│
├── data/                       # Dataset (public_transport.parquet)
├── notebooks/
│   └── clustering_public_transport.ipynb  # Main analysis notebook
├── app/
│   └── app.py                  # Streamlit demo app
├── src/
│   └── preprocess.py           # Data preprocessing functions
│   └── clustering.py           # Clustering algorithms
├── README.md                   # This file
└── report.md                   # Project report
```

## Installation

1. Clone the repository.
2. Install dependencies: `pip install pandas scikit-learn matplotlib streamlit`

## Usage

- Run the notebook: `jupyter notebook notebooks/clustering_public_transport.ipynb`
- Run the app: `streamlit run app/app.py`
- Use the scripts in `src/` for modular analysis.

## Dataset

The dataset is `public_transport.parquet` (115MB), containing passenger counts and stop information. Due to size, it's not uploaded here; download from Kaggle.

## Results

- Identified high-demand zones using K-Means clustering.
- Silhouette score evaluation.
- Visualizations of clusters.

## SDG 11 Impact

Optimizing routes reduces congestion and emissions, promoting sustainable urban mobility.