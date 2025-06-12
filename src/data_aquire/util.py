import os

def data_path(path):
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "../../data", path))