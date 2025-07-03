import matplotlib
from kneed import KneeLocator
from sklearn import metrics
from sklearn.cluster._hdbscan import hdbscan
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


class Clustering:
    def __init__(self, dfs):
        self.dfs = dfs
        self.labels_ = []
        self.models_ = []

    def opt_epsilon(self, df, min_samples, random_state):
        X_2d = PCA(2, random_state=random_state).fit_transform(
            RobustScaler().fit_transform(df)
        )

        k = min_samples
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_2d)
        dist = np.sort(nbrs.kneighbors(X_2d)[0][:, -1])

        knee = KneeLocator(
            range(len(dist)), dist,
            curve="convex", direction="increasing"
        )
        eps = knee.knee_y if knee.knee_y is not None else np.percentile(dist, 90)
        return X_2d, eps

    def _plot_density(self, i, X, db, eps):
        labels = db.labels_
        plt.figure(figsize=(8, 5))
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='Spectral', s=5)
        plt.title(f"DBSCAN Result x{i} (eps={eps:.2f})")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.grid(True)
        plt.show()

    def DBSCAN_clustering(self, X, eps, i, min_samples=10, random_state=42):
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        self._plot_density(i, X, db, eps)

        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        unique_labels = set(labels)
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            xy = X[class_member_mask & core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=14,
            )

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=6,
            )

        plt.title(f"Estimated number of clusters: {n_clusters_}")
        plt.show()

    def run_DBSCAN(self):
        for i, df in enumerate(self.dfs):
            X_2d, epsilon = self.opt_epsilon(df, min_samples=10, random_state=42)
            print(f"\n=== x{i} ===")
            self.DBSCAN_clustering(X_2d, epsilon, i, min_samples=10, random_state=42)

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
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
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

    def run_k_means(self, opt_k=True, input_k=True, subplots=True):
        for i, df in enumerate(self.dfs):
            print(f"\n=== x{i} ===")
            if opt_k:
                self.optimise_k_means(df, 40)
            if input_k:
                k = int(input("k: "))
                self.kmeans_clustering(df)
            if subplots:
                self.kmeans_clustering_subplots(df)

