import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from matplotlib.patches import Patch
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram

class Visualization:

    PLOTS_DIRECTORY = Path(__file__).resolve().parent.parent.parent.joinpath("plots")

    @staticmethod
    def visualize_dataframe(dataframe, title=f"Scatterplot", save=False):
        if dataframe.shape[1] == 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(dataframe.iloc[:, 0], dataframe.iloc[:, 1], color='b', alpha=0.7)
            plt.xlabel('Feature x')
            plt.ylabel('Feature y')
            plt.title(title)
            plt.grid(True)
            if save:
                plt.savefig(Visualization.PLOTS_DIRECTORY.joinpath(title.replace(" ", "_") + ".png"))
            plt.show()

        elif dataframe.shape[1] == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(dataframe.iloc[:, 0], dataframe.iloc[:, 1], dataframe.iloc[:, 2], color='b', alpha=0.7)
            ax.set_xlabel('Feature x')
            ax.set_ylabel('Feature y')
            ax.set_zlabel('Feature z')
            ax.set_title(title)
            if save:
                plt.savefig(Visualization.PLOTS_DIRECTORY.joinpath(title.replace(" ", "_") + ".png"))
            plt.show()

        else:
            print("Datensatz mit weniger als 2 oder mehr als 3 Dimensionen kann nicht visualisiert werden.")

    @staticmethod
    def silhouette_analysis(silhouette_scores, title="Silhouettenanalyse", save=False):
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

        if save:
            plt.savefig(Visualization.PLOTS_DIRECTORY.joinpath(title.replace(" ", "_") + ".png"))

        plt.show()

    @staticmethod
    def elbow_analysis(wcss, k=None, title="Ellbogendiagramm", save=False):
        plt.figure(figsize=(8, 5))
        plt.plot([w[0] for w in wcss], [w[1] for w in wcss], 'bo-')
        plt.xlabel('Anzahl der Cluster')
        plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
        plt.title(title)
        plt.grid(True)

        if k:
            plt.axvline(k, color='red', linestyle='--', label=f'Optimale Clusteranzahl: k={k}')
            plt.legend()

        if save:
            plt.savefig(Visualization.PLOTS_DIRECTORY.joinpath(title.replace(" ", "_") + ".png"))

        plt.show()

    @staticmethod
    def visualize_clusters(dataframe, labels, title="Cluster", colormap="rainbow", save=False):
        colors, color_mapping = Visualization.__color_mapping_clusters(labels, colormap=colormap)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(dataframe.iloc[:, 0], dataframe.iloc[:, 1], c=colors, s=30, edgecolors='k')
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)

        legend = [
            Patch(facecolor=color, edgecolor='k', label=f'{label}')
            for label, color in color_mapping.items()
        ]

        plt.legend(handles=legend, title="Labels", loc='upper right')

        if save:
            plt.savefig(Visualization.PLOTS_DIRECTORY.joinpath(title.replace(" ", "_") + ".png"))
        plt.show()

    @staticmethod
    def visualize_dendrogram(cluster_matrix, title="Dendrogramm", save=False):
        plt.figure(figsize=(10, 7))
        dendrogram(cluster_matrix)
        plt.title(title)
        plt.xlabel('Datenpunkte')
        plt.ylabel('Distanz')
        if save:
            plt.savefig(Visualization.PLOTS_DIRECTORY.joinpath(title.replace(" ", "_") + ".png"))
        plt.show()

    @staticmethod
    def __color_mapping_clusters(labels, colormap="viridis"):
        clusters = np.unique(labels)
        c = len(clusters)
        cmap = cm.get_cmap(colormap)
        color_mapping = {
            cluster: cmap(i / c) for i, cluster in enumerate(clusters)
        }
        return [color_mapping[label] for label in labels], color_mapping
