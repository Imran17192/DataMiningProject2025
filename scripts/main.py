import Paths
import pandas as pd
from scripts.preprocess.ExplorativeDataAnalysis import ExplorativeDataAnalysis
from scripts.preprocess.FeatureEngineering import FeatureEngineering


def load_data():
    df_x = []
    df_x0 = pd.read_json(Paths.X0_DIR)
    df_x1 = pd.read_json(Paths.X1_DIR)
    df_x2 = pd.read_json(Paths.X2_DIR)

    df_x.append(df_x0)
    df_x.append(df_x1)
    df_x.append(df_x2)

    df_ds1 = []
    for p in Paths.DS1_DIR:
        df = pd.read_json(p)
        df_ds1.append(df)

    # DS2 not a pandas datframe so do it later. it is also just  for puzzle

    return df_x, df_ds1



if __name__ == "__main__":
    df_x, df_ds1 = load_data()

    eda_x =  ExplorativeDataAnalysis(df_x)
    eda_x_df = eda_x.compute_eda("x_data_frame", plot=False)

    feature_engineered_x = FeatureEngineering(eda_x_df)
    feature_engineered_x.compute_features()



    print("Hello World")