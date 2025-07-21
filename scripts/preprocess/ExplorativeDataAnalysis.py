import math
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import preprocessing
from sklearn.decomposition import PCA

class ExplorativeDataAnalysis:
    def __init__(self, dfs):
        self.dfs = dfs

    def inspect_dataframe(self, dfs, name):
        print("------------------------------------------------------", name, "------------------------------------------------------")
        i = 1
        for df in dfs:
            print(i,"." , name, "dataframe\n")
            # first 5 lines of the dataframe
            print("head:")
            print(df.head())
            # last 5 lines of the dataframe
            print("tail:")
            print(df.tail())
            # class of the object, index range so how many line, column rang so how many columns
            # detailed descr of the columns, so amount, column name, how many lines aren't null, datatype of datapoint
            # shows memory usage
            print("info:")
            print(df.info())
            # gives us for each column count of lines, mean, std, min, ...
            print("describe:")
            print(df.describe())
            # checks for all columns if how many null values
            print("isNull.sum")
            print(df.isnull().sum())
            # tells the amount of unique values in each column
            print("nunique")
            print(df.nunique())
            i += 1


    def remove_outliers(self, dfs):
        df_cleaned = []
        for df in dfs:
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)

            IQR = Q3 - Q1

            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            df_no_outliers = df[((df >= lower_limit) & (df <= upper_limit)).all(axis=1)]
            df_cleaned.append(df_no_outliers)

        return df_cleaned


    def dfs_plot(self, dfs):
        for df in dfs:
            df.plot()

            plt.show()


    def standardize_df(self, dfs):
        dfs_std = []
        for df in dfs:
            df_std = (df - df.mean()) / df.std()
            dfs_std.append(df_std)
        return dfs_std


    def normalize_df(self, dfs):
        dfs_norm = []
        for df in dfs:
            df_norm = (df - df.min()) / (df.max() - df.min())
            dfs_norm.append(df_norm)
        return dfs_norm


    def plot_bar(self, dfs):
        for df in dfs:
            # hist creates plot for all columns of df columns and bins for amount of pillars
            df.hist(bins=100, figsize=(15, 10))
            plt.suptitle("Histogram plot of dataframe")
            plt.show()

    def plot_kernel(self, dfs):
        for df in dfs:
            D = df.shape[1]
            square_root = math.ceil(D ** 0.5)
            fig, axes = plt.subplots(nrows=square_root, ncols=square_root, figsize=(15, 10), constrained_layout=True)
            # flatten axes for easier iteration
            axes = axes.flatten()
            for i in range(D):
                sns.kdeplot(df.iloc[:, i], ax=axes[i])
                axes[i].set_title(f'Distribution of {df.columns[i]}')
            plt.show()

    def plot_heat(self, dfs):
        for df in dfs:
            corr = df.corr()
            plt.figure(figsize=(15, 10))
            sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, linecolor="white")
            plt.show()

    def plot_index_scatter(self, dfs):
        for df in dfs:
            D = df.shape[1]
            square_root = math.ceil(D ** 0.5)
            fig, axes = plt.subplots(nrows=square_root, ncols=square_root,
                                     figsize=(15, 10), constrained_layout=True)
            axes = axes.flatten()
            for i in range(D):
                axes[i].scatter(df.index, df.iloc[:, i], alpha=0.5,
                                 marker='x', s=10)
                axes[i].set_title(f'Feature {df.columns[i]}')
                axes[i].set_xlabel('Index')
                axes[i].set_ylabel('Value')
            plt.show()

    def pca_reduction(self, threshold = 0.95, dim_redu_count=1,show_plots: bool = True  ):
        dfs_pca = []
        for df in self.dfs:
            scaled_data = preprocessing.scale(df.T)
            pca = PCA()
            pca.fit(scaled_data)

            cum_var = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.searchsorted(cum_var, threshold) + 1

            orig_dim = df.shape[1]
            use_pca = (orig_dim - n_components) >= dim_redu_count
            if use_pca:
                if show_plots:
                    components = np.arange(1, len(cum_var) + 1)
                    plt.figure(figsize=(10, 4))
                    plt.plot(components, cum_var,
                             marker='o', linestyle='--')
                    plt.axhline(y=threshold, color='r', linestyle='-')
                    plt.text(0.5, threshold - 0.05,
                             f'{int(threshold * 100)} %-Threshold',
                             color='red')
                    plt.xlabel('Number of components')
                    plt.ylabel('accumulated variance')
                    plt.title('accumulative explained variance')
                    plt.tight_layout()
                    plt.show()
                per_var = np.round(pca.explained_variance_ratio_ * 100, 1)
                labels = [f'PC{i}' for i in range(1, len(per_var) + 1)]

                if show_plots:
                    plt.bar(x=labels, height=per_var)
                    plt.ylabel('Variance in %')
                    plt.xlabel('PC')
                    plt.title('Scree-Plot')
                    plt.tight_layout()
                    plt.show()

                pca_data = pca.transform(scaled_data)
                pca_df = pd.DataFrame(
                    pca_data,
                    index=df.columns,
                    columns=labels
                )

                if show_plots:
                    plt.scatter(pca_df["PC1"], pca_df["PC2"])
                    plt.title('PCA-Scatter')
                    plt.xlabel(f"PC1 – {per_var[0]} %")
                    plt.ylabel(f"PC2 – {per_var[1]} %")
                    for sample in pca_df.index:
                        plt.annotate(sample,
                                     (pca_df.loc[sample, "PC1"],
                                      pca_df.loc[sample, "PC2"]))
                    plt.tight_layout()
                    plt.show()

                reduced = pca_df.iloc[:, :n_components]
                dfs_pca.append(reduced.T)

            else:
                dfs_pca.append(df)

        return dfs_pca


    def compute_eda(self, name, plot = False, clean=False, classify = False):
        df = self.dfs

        if classify:
            self.cut_inner_ring(df)
            self.remove_outliers(df)
            self.standardize_df(df)
            self.pca_reduction(df)
            self.split_data(df)

        else:

            if clean:
                df = self.standardize_df(df)

            self.inspect_dataframe(df, name)

            if plot:
                self.plot_bar(df)
                self.plot_kernel(df)
                self.plot_heat(df)
                self.plot_index_scatter(df)

            return df
        #-------------------------------------------------------------------
