import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from scipy.cluster.hierarchy import dendrogram

class Visualization:

    @staticmethod
    def visualize_dataframe(dataframe):
        if dataframe.shape[1] == 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(dataframe.iloc[:, 0], dataframe.iloc[:, 1], color='b', alpha=0.7)
            plt.xlabel('Feature x')
            plt.ylabel('Feature y')
            plt.title('Visualisierung')
            plt.grid(True)
            plt.show()

        elif dataframe.shape[1] == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(dataframe.iloc[:, 0], dataframe.iloc[:, 1], dataframe.iloc[:, 2], color='b', alpha=0.7)
            ax.set_xlabel('Feature x')
            ax.set_ylabel('Feature y')
            ax.set_zlabel('Feature z')
            ax.set_title('Visualisierung')
            plt.show()

        else:
            print("Datensatz mit weniger als 2 oder mehr als 3 Dimensionen kann nicht visualisiert werden.")

    @staticmethod
    def silhouette_analysis(silhouette_scores, title="Silhouettenanalyse"):
        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot([silhouette_score[0] for silhouette_score in silhouette_scores], [silhouette_score[1] for silhouette_score in silhouette_scores], marker='o')
        plt.title(title)
        plt.xlabel("Anzahl der Cluster")
        plt.ylabel("Silhouettenkoeffizient")
        plt.grid(True)

        # Optional: Beste Anzahl hervorheben
        best = max(silhouette_scores, key=lambda x: x[1])
        best_c = best[0]
        plt.axvline(best_c, color='red', linestyle='--', label=f'Optimale Clusteranzahl: c={best_c}')
        plt.legend()

        plt.show()

    @staticmethod
    def visualize_clusters(dataframe, labels, title="Cluster", colormap="rainbow"):
        colors = Visualization.__color_mapping_clusters(labels, colormap=colormap)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(dataframe.iloc[:, 0], dataframe.iloc[:, 1], c=colors, s=30, edgecolors='k')
        plt.title(title)
        plt.xlabel(dataframe.columns[0])
        plt.ylabel(dataframe.columns[1])
        plt.grid(True)
        plt.colorbar(scatter, label="Cluster")
        plt.show()

    @staticmethod
    def visualize_dendrogram(cluster_matrix, title="Dendrogramm"):
        plt.figure(figsize=(10, 7))
        dendrogram(cluster_matrix)
        plt.title(title)
        plt.xlabel('Datenpunkte')
        plt.ylabel('Distanz')
        plt.show()

    @staticmethod
    def __color_mapping_clusters(labels, colormap="viridis"):
        clusters = np.unique(labels)
        c = len(clusters)
        cmap = cm.get_cmap(colormap)
        color_mapping = {
            cluster: cmap(i / c) for i, cluster in enumerate(clusters)
        }
        return [color_mapping[label] for label in labels]
