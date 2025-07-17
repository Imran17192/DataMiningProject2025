import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Preprocessing_Classify:
    def __init__(self, dfs, y, x_test):
        self.dfs = dfs
        self.y = y
        self.x_test = x_test

    def standardize_df(self, df):
        return (df - df.mean()) / df.std()

    def normalize_df(self, df):
        return (df - df.min()) / (df.max() - df.min())

    def remove_outliers_with_labels(self, df, y):
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1

        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        mask = ((df >= lower_limit) & (df <= upper_limit)).all(axis=1)

        df_no_outliers = df[mask]
        y_filtered = y[mask.values]

        return df_no_outliers, y_filtered

    def remove_outliers(self, df):
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1

        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        mask = ((df >= lower_limit) & (df <= upper_limit)).all(axis=1)
        return df[mask]

    def pca_reduction(self, df, threshold=0.9, dim_redu_count=1, show_plots=False):
        df = df.dropna()
        scaled_data = preprocessing.scale(df)
        pca = PCA()
        pca.fit(scaled_data)

        cum_var = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.searchsorted(cum_var, threshold) + 1

        orig_dim = df.shape[1]
        use_pca = (orig_dim - n_components) >= dim_redu_count

        per_var = np.round(pca.explained_variance_ratio_ * 100, 1)
        labels = [f'PC{i}' for i in range(1, len(per_var) + 1)]

        if show_plots:
            plt.figure(figsize=(10, 4))
            plt.plot(np.arange(1, len(cum_var) + 1), cum_var, marker='o', linestyle='--')
            plt.axhline(y=threshold, color='r', linestyle='-')
            plt.title('accumulative explained variance')
            plt.tight_layout()
            plt.show()

            plt.bar(x=labels, height=per_var)
            plt.title('Scree-Plot')
            plt.tight_layout()
            plt.show()

        pca_data = pca.transform(scaled_data)
        pca_df = pd.DataFrame(pca_data, index=df.index, columns=labels)

        if show_plots and "PC1" in pca_df and "PC2" in pca_df:
            plt.scatter(pca_df["PC1"], pca_df["PC2"])
            plt.title('PCA-Scatter')
            plt.xlabel(f"PC1 – {per_var[0]} %")
            plt.ylabel(f"PC2 – {per_var[1]} %")
            plt.tight_layout()
            plt.show()

        reduced = pca_df.iloc[:, :n_components]
        return reduced

    def split_data(self, df, y, test_size=0.3, random_state=42):
        x_train, x_valid, y_train, y_valid = train_test_split(
            df, y, test_size=test_size, random_state=random_state
        )
        return x_train, x_valid, y_train, y_valid

    def compute_eda(self):
        X_train_list = []
        X_valid_list = []
        y_train_list = []
        y_valid_list = []

        for i, df in enumerate(self.dfs):
            y_series = pd.Series(self.y[i])

            df, y_series = self.remove_outliers_with_labels(df, y_series)
            df = df.fillna(df.mean())
            df = self.standardize_df(df)
            df = self.pca_reduction(df, threshold=1.0, show_plots=False)

            X_train, X_valid, y_train, y_valid = self.split_data(df, y_series)
            X_train_list.append(X_train)
            X_valid_list.append(X_valid)
            y_train_list.append(y_train)
            y_valid_list.append(y_valid)


        x_test_processed = self.x_test.copy()
        x_test_processed = self.remove_outliers(x_test_processed)
        x_test_processed = x_test_processed.fillna(x_test_processed.mean())
        x_test_processed = self.standardize_df(x_test_processed)
        x_test_processed = self.pca_reduction(x_test_processed, threshold=1.0, show_plots=False)

        return X_train_list, X_valid_list, y_train_list, y_valid_list, x_test_processed
