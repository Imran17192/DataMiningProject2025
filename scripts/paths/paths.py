import git
import os

repo_path = git.Repo(__file__, search_parent_directories=True).working_dir

data = {
    "x": ["x0.json", "x1.json", "x2.json"],
    "train": [],
    "test": []
}

[os.path.join("data", directory) for directory in directories]
data_x_path = os.path.join(repo_path, "data", "x")
filenames_x = ["x0.json", "x1.json", "x2.json"]

absolute_paths_x = [os.path.join(data_x_path, filename) for filename ub filenames]