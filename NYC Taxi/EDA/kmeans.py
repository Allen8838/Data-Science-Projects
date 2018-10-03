import numpy as np 
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt


def find_kmeans_clusters_graph(df, coords, lat1, long1, lat2, long2):
    sample_ind = np.random.permutation(len(coords))[:500000]
    kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])

    df['pickup_cluster'] = kmeans.predict(df[['pickup_latitude', 'pickup_longitude']])
    df['dropoff_cluster'] = kmeans.predict(df[['dropoff_latitude', 'dropoff_longitude']])
    
    N = 100000
    city_long_border = (-74.03, -73.75)
    city_lat_border = (40.63, 40.85)
    plt.figure(figsize=(7,7))
    plt.scatter(df.pickup_longitude.values[:N], df.pickup_latitude.values[:N], s=10, lw=0,
           c=df.pickup_cluster[:N].values, cmap='tab20', alpha=0.2)
    _ = plt.xlim(city_long_border)
    _ = plt.ylim(city_lat_border)

    plt.savefig('kmeans_clusters.png')

    return None