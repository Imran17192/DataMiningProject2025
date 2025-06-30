import math
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer




class FeatureEngineering:
    def __init__(self, dfs):
        self.dfs = dfs

    def normalize_df(self, dfs):
        dfs_norm = []
        for df in dfs:
            df_norm = (df - df.min()) / (df.max() - df.min())
            dfs_norm.append(df_norm)
        return dfs_norm

    def bins_sturges(self, n):
        return int(np.ceil(np.log2(n)) + 1)

    def discretise_df(self):
        dfs_disc = []
        for df in self.dfs:
            number_bins = self.bins_sturges(df.shape[0])
            equal_width_discretizer = KBinsDiscretizer(n_bins= number_bins, encode='ordinal', strategy='uniform')
            num_cols = df.select_dtypes(include=[np.number]).columns
            df_disc = df.copy()
            df_disc[num_cols] = equal_width_discretizer.fit_transform(df[num_cols])
            dfs_disc.append(df_disc)
        return dfs_disc

    def compute_features(self, threshold = 0.9, dim_redu_count=1,show_plots: bool = True  ):
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
