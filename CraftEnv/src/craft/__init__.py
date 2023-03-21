import os


def get_data_path():
    return os.path.join(os.path.dirname(__file__), "data")


def get_urdf_path():
    return os.path.join(get_data_path(), "urdf")
