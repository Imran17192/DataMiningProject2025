import json

from sklearn.model_selection import train_test_split
import os  # ← NEU
import numpy as np  # ← NEU
from scipy.stats import mode  # ← NEU

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

    # DS2 not a pandas datframe so do it later. it is also just  for puzzle

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


def dm_part3(df_x, labels, x_test):

    prepare = Preprocessing_Classify(df_x, labels, x_test)
    X_train_list, X_valid_list, y_train_list, y_valid_list, x_test_processed = prepare.compute_eda()

    svm = Classify(X_train_list[0], y_train_list[0], X_valid_list[0], y_valid_list[0], model_type="svm")
    svm.train_svm()

    lr = Classify(X_train_list[0], y_train_list[0], X_valid_list[0], y_valid_list[0], model_type="logreg")
    lr.train_logreg()

    gnb = Classify(X_train_list[0], y_train_list[0], X_valid_list[0], y_valid_list[0], model_type="gnb")
    gnb.train_gnb()

    knn = Classify(X_train_list[0], y_train_list[0], X_valid_list[0], y_valid_list[0], model_type="knn")
    knn.train_knn()

    # TODO take hard average of all prediciton for final prediciotn json perhaps visualize predictions


    return


if __name__ == "__main__":
    df_x, df_ds1 = load_data()

    # df_x_preprocessed, df_ds1_preprocessed = dm_part1(df_x, df_ds1)

    # dm_part2(df_x_preprocessed, df_ds1_preprocessed)

    labels_y = load_labels()
    x_test = load_test()

    dm_part3(df_x, labels_y, x_test)
