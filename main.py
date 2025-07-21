import json

from sklearn.model_selection import train_test_split
import os
import numpy as np
from scipy.stats import mode

import paths
import pandas as pd

from scripts.preprocess.ExplorativeDataAnalysis import ExplorativeDataAnalysis
from scripts.preprocess.FeatureEngineering import FeatureEngineering
from scripts.preprocess.Preprocessing import Preprocessing
from scripts.train.HierarchicalClustering import HierarchicalClustering
from scripts.train.kMeans import kMeans
from scripts.unsupervised_learning.Clustering import Clustering
from scripts.visualization.Visualization import Visualization

from scripts.preprocess.Preprocessing_Classify import Preprocessing_Classify
from scripts.classification.Classifiy import Classify
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def load_data():
    df_x = []
    df_x0 = pd.read_json(paths.X0_DIR)
    df_x1 = pd.read_json(paths.X1_DIR)
    df_x2 = pd.read_json(paths.X2_DIR)

    df_x.append(df_x0)
    df_x.append(df_x1)
    df_x.append(df_x2)

    df_ds1 = []
    for p in paths.DS1_DIR:
        df = pd.read_json(p)
        df_ds1.append(df)
    return df_x, df_ds1


def load_labels():
    labels = []
    for labels_path in [paths.Y0_DIR, paths.Y1_DIR, paths.Y2_DIR]:
        with open(labels_path) as labels_file:
            y = json.load(labels_file)
            labels.append(y)
    return labels


def load_test():
    return pd.read_json(paths.X_TEST)


def dm_part1(df_x, df_ds1):
    eda_x = ExplorativeDataAnalysis(df_x)
    eda_x_df = eda_x.compute_eda("x_data_frame", plot=False, clean=True)

    eda_da1 = ExplorativeDataAnalysis(df_ds1)
    eda_da1_df = eda_da1.compute_eda("x_data_frame", plot=False, clean=True)

    feature_engineered_x = FeatureEngineering(eda_x_df)
    pca_x = feature_engineered_x.compute_features(show_plots=False)

    feature_engineered_ds1 = FeatureEngineering(eda_da1_df)
    pca_ds1 = feature_engineered_ds1.compute_features(show_plots=False)

    return eda_x_df, eda_da1_df


def dm_part2(df1, df2):
    hierarchical_clustering = Clustering(df1)
    kmean = Clustering(df1)
    dbscan = Clustering(df1)

    for linkage_method in ["single", "complete", "average"]:
        hierarchical_clustering.silhouette_analysis(linkage_method)
        hierarchical_clustering.linkage_clustering(linkage_method)

    hierarchical_clustering_results = hierarchical_clustering.get_clustering_results()
    for clustering_method in hierarchical_clustering_results:
        for dataframe in hierarchical_clustering_results[clustering_method]:
            Visualization.visualize_clusters(
                hierarchical_clustering_results[clustering_method][dataframe]["dataframe"],
                hierarchical_clustering_results[clustering_method][dataframe]["labels"],
                title=f"Clusteringergebnis {clustering_method}-Linkage ({dataframe}"
            )

    kmean.run_k_means(opt_k=True, input_k=True, subplots=False, evaluate=False)
    kmean.run_k_means(opt_k=False, input_k=False, subplots=False, evaluate=False)
    dbscan.run_DBSCAN(fast_compute=False)
    for i, df in enumerate(df1):
        print(f"\n=== Erweiterte Clusterverfahren für df{i} ===")
        kmean.fuzzy_c_means(df, c=3)
        kmean.mini_batch_kmeans(df, k=3)
        kmean.em_gaussian_mixture(df, n_components=3)

def plot_label_distribution(labels, title="Label‑Verteilung"):
    vc = pd.Series(labels).value_counts().sort_index()
    plt.figure(figsize=(10, 4))
    sns.barplot(x=vc.index, y=vc.values, palette="tab20")
    plt.xlabel("Label")
    plt.ylabel("Anzahl")
    plt.tight_layout()
    plt.show()


def plot_label_feature_heatmap(X, labels, title="Heatmap"):
    df = X.copy()
    df["label"] = labels
    mean_per_label = df.groupby("label").mean().sort_index()
    sns.heatmap(mean_per_label, cmap="viridis", cbar_kws=dict(label="Feature‑Mittelwert"))
    plt.title(title)
    plt.ylabel("Label‑Klasse")
    plt.xlabel("Feature")
    plt.tight_layout()
    plt.show()


def plot_label_gallery_single(df, labels):
    print("hello")
    df = df.copy()
    df["label"] = labels

    label_ids = sorted(df["label"].unique())
    num_labels = len(label_ids)

    fig, axs = plt.subplots(nrows=(num_labels + 4) // 5, ncols=5, figsize=(15, num_labels // 2))
    axs = axs.flatten()

    for i, label in enumerate(label_ids):
        sample = df[df["label"] == label].iloc[0, :-1].values
        side = int(np.sqrt(len(sample)))
        matrix = sample[:side * side].reshape(side, side)

        sns.heatmap(matrix, ax=axs[i], cmap="viridis", cbar=False,
                    xticklabels=False, yticklabels=False)
        axs[i].set_title(f"Label {label}", fontsize=9)

    for ax in axs[num_labels:]:
        ax.axis("off")

    plt.suptitle("Heatmap pro Klasse – 1 Beispiel je Label", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def quadratic_heatmap_dataset_two(df_x, labels, dataset_name="Dataset"):
    df = df_x.copy()
    df["label"] = labels

    selected_labels = [5, 20, 29]

    for lbl in selected_labels:
        df_label = df[df["label"] == lbl]
        n = min(len(df_label), 8)
        if n == 0:
            print(f"Keine Objekte mit Label {lbl} in {dataset_name}.")
            continue

        fig, axs = plt.subplots(2, 4, figsize=(20, 6))
        axs = axs.flatten()

        for i in range(n):
            feature_vector = df_label.iloc[i, :-1].values
            side = int(np.sqrt(feature_vector.size))
            matrix = feature_vector[:side*side].reshape(side, side)

            sns.heatmap(
                matrix,
                cmap="viridis",
                cbar=True,
                xticklabels=False, yticklabels=False,
                ax=axs[i]
            )
            axs[i].set_title(f"{dataset_name} - Label {lbl}, Index {df_label.index[i]}")
            axs[i].set_xlabel(f"{side}×{side}")

        for ax in axs[n:]:
            ax.axis("off")

        fig.suptitle(f"Heatmaps der ersten Objekte mit Label {lbl} ({dataset_name})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    plot_label_gallery_single(df, labels)

def dm_part3(df_x, labels, x_test):
    for i in range(0,3):
        quadratic_heatmap_dataset_two(df_x[i], labels[i])

    prepare = Preprocessing_Classify(df_x, labels, x_test)
    X_train_list, X_valid_list, y_train_list, y_valid_list, x_test_processed = prepare.compute_eda()

    X_full = pd.concat([X_train_list[0], X_valid_list[0]], ignore_index=True)
    y_full = np.concatenate([y_train_list[0], y_valid_list[0]])

    plot_label_distribution(y_full, "Label‑Verteilung (Trainingalid)")
    plot_label_feature_heatmap(X_full, y_full, "Heatmap der durchschnittlichen Feature‑Werte pro Label (Train+Valid)")

    y_test_preds = []

    svm = Classify(X_train_list[0], y_train_list[0], X_valid_list[0], y_valid_list[0], model_type="svm")
    svm.train_svm()
    y_test_preds.append(svm.model.predict(x_test_processed))

    lr = Classify(X_train_list[0], y_train_list[0], X_valid_list[0], y_valid_list[0], model_type="logreg")
    lr.train_logreg()
    y_test_preds.append(lr.model.predict(x_test_processed))

    gnb = Classify(X_train_list[0], y_train_list[0], X_valid_list[0], y_valid_list[0], model_type="gnb")
    gnb.train_gnb()
    y_test_preds.append(gnb.model.predict(x_test_processed))

    knn = Classify(X_train_list[0], y_train_list[0], X_valid_list[0], y_valid_list[0], model_type="knn")
    knn.train_knn()
    y_test_preds.append(knn.model.predict(x_test_processed))

    np_y_test_preds = np.array(y_test_preds)
    most_frequent_y_preds = mode(np_y_test_preds, axis=0, keepdims=False).mode

    Classify.save_predictions(most_frequent_y_preds, path="predictions/average_prediction.json")
    plot_label_distribution(most_frequent_y_preds, "Label‑Verteilung (Ensemble‑Predictions)")

    quadratic_heatmap_dataset_two(x_test_processed, most_frequent_y_preds)


if __name__ == "__main__":
    df_x, df_ds1 = load_data()

    # df_x_preprocessed, df_ds1_preprocessed = dm_part1(df_x, df_ds1)

    # dm_part2(df_x_preprocessed, df_ds1_preprocessed)

    labels_y = load_labels()
    x_test = load_test()

    dm_part3(df_x, labels_y, x_test)
