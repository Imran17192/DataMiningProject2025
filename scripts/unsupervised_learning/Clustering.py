import matplotlib
from sklearn.decomposition import PCA

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class Clustering:
    def __init__(self, dfs):
        self.dfs = dfs
        self.labels_ = []
        self.models_ = []

    def optimise_k_means(self, df, max_k=10):
        inertias = []
        k_range = range(1, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df)
            inertias.append(kmeans.inertia_)

        plt.figure(figsize=(10, 5))
        plt.plot(k_range, inertias, marker='o')
        plt.title('Elbow Curve')
        plt.xlabel('Anzahl der Cluster (k)')
        plt.ylabel('Inertia')
        plt.grid(True)
        plt.show()

    def kmeans_clustering(self, df, k=3):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(df)

        self.models_.append(kmeans)
        self.labels_.append(labels)

        score = silhouette_score(df, labels)
        print(f"Silhouette Score (k={k}): {score:.3f}")

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(df)

        plt.figure(figsize=(20, 30))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=10)
        plt.title(f'KMeans Clustering (k={k})')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.grid(True)
        plt.show()

    def kmeans_clustering_subplots(self, df):
        fig, axes = plt.subplots(3,3 , figsize=(18, 15))
        axes = np.array(axes).flatten()

        for i, k in enumerate(range(2, 11)):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(df)

            pca = PCA(n_components=2)
            reduced = pca.fit_transform(df)

            axes[i].scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=10)
            axes[i].set_title(f'KMeans Clustering (k={k})')
            axes[i].set_xlabel('PCA 1')
            axes[i].set_ylabel('PCA 2')

        plt.tight_layout()
        plt.show()


    def run_k_means(self, opt_k=True,input_k=True,subplots=True):
        for i, df in enumerate(self.dfs):
            print(f"\n=== x{i} ===")
            if opt_k:
                self.optimise_k_means(df, 40)
            if input_k:
                k = int(input("k: "))
                self.kmeans_clustering(df)
            if subplots:
                self.kmeans_clustering_subplots(df)

