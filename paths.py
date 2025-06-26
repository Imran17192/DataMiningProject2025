from pathlib import Path

def iterate_folder_json_paths(folder_path):
    DS_paths = []
    for p in folder_path.iterdir():
        if p.is_file() and p.suffix == '.json':
            DS_paths.append(p)
        p.joinpath(".json")
    return DS_paths

# Root folder path
BASE_DIR = Path(__file__).resolve().parent.parent.joinpath('DataMiningProject2025')

# Path to all main folders
DATA_DIR = BASE_DIR.joinpath('data')
MODEL_DIR = BASE_DIR.joinpath('models')
PLOTS_DIR = BASE_DIR.joinpath('plots')
PREDICTIONS_DIR = BASE_DIR.joinpath('predictions')
SCORES_DIR = BASE_DIR.joinpath('scores')

# Path to raw data
X_DIR = DATA_DIR.joinpath('x')

X0_DIR = X_DIR.joinpath('x0.json')
X1_DIR = X_DIR.joinpath('x1.json')
X2_DIR = X_DIR.joinpath('x2.json')

# Path to additional data
DS1_DIR = iterate_folder_json_paths(X_DIR.joinpath('12_8_2_12_13'))
DS2_DIR = X_DIR.joinpath("13_8_5_19_8_12").joinpath("13_8_5_19_8_12").joinpath('.json')
