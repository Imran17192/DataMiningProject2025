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
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
import skfuzzy as fuzz


from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


class Clustering:
    def __init__(self, dfs):
        self.dfs = dfs
        self.labels_ = []
        self.models_ = []

    def DBSCAN_evaluation(self, df, eps, min_samples):
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(df)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"Anzahl Cluster: {n_clusters}")
        print(f"Anzahl Rauschen (Noise): {list(labels).count(-1)}")

        sil = silhouette_score(df, labels)
        db = davies_bouldin_score(df, labels)
        ch = calinski_harabasz_score(df, labels)
        print(f"Silhouette Score: {sil:.3f}")
        print(f"Davies-Bouldin Index: {db:.3f}")
        print(f"Calinski-Harabasz Score: {ch:.3f}")


    def plot_k_distance_graph(self, X):
        k = X.shape[1] * 2
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(X)
        distances, _ = neigh.kneighbors(X)
        distances = np.sort(distances[:, k - 1])

        kneedle = KneeLocator(
            range(len(distances)), distances,
            curve='convex', direction='increasing'
        )
        best_eps = distances[kneedle.knee] if kneedle.knee is not None else None

        plt.figure(figsize=(10, 6))
        plt.plot(distances, label='k-distances')
        if best_eps is not None:
            plt.axvline(kneedle.knee, color='red', linestyle='--', label=f'Best eps ≈ {best_eps:.3f}')

        plt.xlabel('Points')
        plt.ylabel(f'{k}-th nearest neighbor distance')
        plt.title('K-distance Graph')
        plt.show()

        print("best eps", best_eps)

        return best_eps, k

    def run_DBSCAN(self, fast_compute=False):
        for i, df in enumerate(self.dfs):
            if fast_compute:
                epsilon = 1.0
                min_samples = 5
            else:
                epsilon, min_samples = self.plot_k_distance_graph(df)
            print(f"\n=== x{i} ===")
            self.DBSCAN_evaluation(df,epsilon,min_samples)

    def optimise_k_means(self, df, max_k=10):
        inertias = []
        k_range = range(1, 30)

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

        kn = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
        best_k = kn.knee

        return best_k

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

    def kmeans_evaluation(self, df, name):
        inertia = {0: "Inertia (Elbow Method)"}
        silhouette = {0: "Silhouette Coefficient"}
        davies = {0: "Davies-Bouldin Index"}
        calhar = {0: "Calinski-Harabasz Score"}
        eval_scores = [inertia, silhouette, davies, calhar]

        k_range = range(2, 100)

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42).fit(df)
            labels = kmeans.labels_

            inertia[k] = kmeans.inertia_
            silhouette[k] = silhouette_score(df, labels)
            davies[k] = davies_bouldin_score(df, labels)
            calhar[k] = calinski_harabasz_score(df, labels)

        for element in eval_scores:
            print(element)

        for j, score_dict in enumerate(eval_scores):
            title = score_dict.pop(0)
            x = list(score_dict.keys())
            y = list(score_dict.values())
            axes[j].plot(x, y, marker='o')
            axes[j].set_title(title)
            axes[j].set_xlabel("k")
            axes[j].set_ylabel("Score")
            score_dict[0] = title

        plt.suptitle(f"KMeans Evaluation: {name}", fontsize=14)
        plt.tight_layout()
        plt.show()


    def run_k_means(self, opt_k=True, input_k=True, subplots=True, evaluate=True):
        for i, df in enumerate(self.dfs):
            print(f"\n=== x{i} ===")
            if opt_k:
                k = self.optimise_k_means(df, 30)
            if input_k:
                self.kmeans_clustering(df,k)
            if subplots:
                self.kmeans_clustering_subplots(df)
            if evaluate:
                self.kmeans_evaluation(df, i)

    def fuzzy_c_means(self, df, c=3, m=2.0):
        data = df.to_numpy().T  # shape: (features, samples)
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            data, c=c, m=m, error=0.005, maxiter=1000, init=None
        )

        labels = np.argmax(u, axis=0)

        print(f"Fuzzy C-Means: {c} Cluster")
        sil = silhouette_score(df, labels)
        print(f"Silhouette Score: {sil:.3f}")

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(df)

        plt.figure(figsize=(10, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=10)
        plt.title(f'Fuzzy C-Means Clustering (c={c})')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.grid(True)
        plt.show()

    def mini_batch_kmeans(self, df, k=3):
        model = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=100)
        labels = model.fit_predict(df)

        print(f"MiniBatchKMeans: {k} Cluster")
        sil = silhouette_score(df, labels)
        print(f"Silhouette Score: {sil:.3f}")

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(df)

        plt.figure(figsize=(10, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=10)
        plt.title(f'MiniBatchKMeans Clustering (k={k})')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.grid(True)
        plt.show()

    def hierarchical_clustering(self, df, n_clusters=3, linkage='ward'):
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = model.fit_predict(df)

        print(f"Hierarchisches Clustering: {n_clusters} Cluster (Linkage: {linkage})")
        sil = silhouette_score(df, labels)
        print(f"Silhouette Score: {sil:.3f}")

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(df)

        plt.figure(figsize=(10, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=10)
        plt.title(f'Agglomerative Clustering (n={n_clusters})')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.grid(True)
        plt.show()
