import yaml
from easydict import EasyDict as edict
"""
The configuration file. You can config the file paths.
"""
def get_config(config_path="config.yaml"):
    """
    Get the configurations.
    read the yaml file identified by config_path
    """
    with open(config_path, "r") as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    return cfg