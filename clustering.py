# clustering.py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

def find_optimal_clusters(data, max_k=10):
    distortions = []
    for i in range(1, max_k):
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(data)
        distortions.append(km.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()

def perform_clustering(data, n_clusters=3):
    # KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(data)
    
    # Agglomerative Clustering
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    agglomerative_labels = agglomerative.fit_predict(data)
    
    print(f"KMeans Silhouette Score: {silhouette_score(data, kmeans_labels)}")
    print(f"Agglomerative Silhouette Score: {silhouette_score(data, agglomerative_labels)}")
