import pandas as pd

def load_data(filepath):
    """Load the public transport parquet file."""
    return pd.read_parquet(filepath)

def preprocess_data(df):
    """Preprocess the data: select columns, remove outliers, aggregate by stop_id."""
    # Keep relevant columns
    cols = ['stop_id', 'passengers']
    df = df[cols].dropna()

    # Remove outliers or unrealistic values
    df = df[df['passengers'] > 0]

    # Aggregate by stop_id
    df_grouped = df.groupby('stop_id').agg(
        avg_passengers=('passengers', 'mean'),
        total_passengers=('passengers', 'sum')
    ).reset_index()

    return df_grouped

if __name__ == "__main__":
    df = load_data("../data/public_transport.parquet")
    df_processed = preprocess_data(df)
    print("Processed data shape:", df_processed.shape)
    print(df_processed.head())