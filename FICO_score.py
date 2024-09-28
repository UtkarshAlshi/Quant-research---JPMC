import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def load_data():
    return pd.DataFrame({
        'fico_score': [580, 620, 650, 700, 720, 760, 780, 800, 820, 840],
        'default':    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
    })

def quantize_fico_scores(df, n_buckets):
    kmeans = KMeans(n_clusters=n_buckets, random_state=42)
    df['bucket'] = kmeans.fit_predict(df[['fico_score']])
    bucket_centers = df.groupby('bucket')['fico_score'].mean().sort_values()
    return df, bucket_centers

def calculate_mse(df):
    bucket_means = df.groupby('bucket')['fico_score'].transform('mean')
    return mean_squared_error(df['fico_score'], bucket_means)

def calculate_log_likelihood(df):
    grouped = df.groupby('bucket').agg({'default': ['sum', 'count']})
    grouped.columns = ['defaults', 'total']
    
    p = grouped['defaults'] / grouped['total']
    p = np.clip(p, 1e-10, 1 - 1e-10)  # Avoid log(0)
    
    return np.sum(grouped['defaults'] * np.log(p) + 
                  (grouped['total'] - grouped['defaults']) * np.log(1 - p))

def plot_buckets(df, bucket_centers):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['fico_score'], df['bucket'], c=df['bucket'], cmap='rainbow')
    plt.plot(bucket_centers, range(len(bucket_centers)), 'ro', markersize=10)
    plt.title('FICO Score Buckets')
    plt.xlabel('FICO Score')
    plt.ylabel('Bucket')
    plt.colorbar(label='Bucket')
    plt.show()

def main():
    df = load_data()
    n_buckets = 3
    df, bucket_centers = quantize_fico_scores(df, n_buckets)
    
    mse = calculate_mse(df)
    log_likelihood = calculate_log_likelihood(df)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Log-Likelihood: {log_likelihood:.4f}")
    
    plot_buckets(df, bucket_centers)

if __name__ == "__main__":
    main()