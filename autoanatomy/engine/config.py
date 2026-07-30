import os
import json
from pathlib import Path
import importlib.metadata
import importlib.resources


def get_totalseg_dir():
    if "AUTOANATOMY_HOME_DIR" in os.environ:
        totalseg_dir = Path(os.environ["AUTOANATOMY_HOME_DIR"])
    else:
        # in docker container finding home not properly working therefore map to /tmp
        home_path = Path("/tmp") if str(Path.home()) == "/" else Path.home()
        totalseg_dir = home_path / ".autoanatomy"
    return totalseg_dir


def get_weights_dir():
    if "AUTOANATOMY_WEIGHTS_PATH" in os.environ:
        # config_dir = Path(os.environ["AUTOANATOMY_WEIGHTS_PATH"]) / "nnUNet"
        config_dir = Path(os.environ["AUTOANATOMY_WEIGHTS_PATH"])
    else:
        totalseg_dir = get_totalseg_dir()
        config_dir = totalseg_dir / "nnunet/results"
    return config_dir


def setup_nnunet():
    # check if environment variable AUTOANATOMY_WEIGHTS_PATH is set
    if "AUTOANATOMY_WEIGHTS_PATH" in os.environ:
        weights_dir = os.environ["AUTOANATOMY_WEIGHTS_PATH"]
    else:
        # in docker container finding home not properly working therefore map to /tmp
        config_dir = get_totalseg_dir()
        # (config_dir / "nnunet/results/nnUNet/3d_fullres").mkdir(exist_ok=True, parents=True)
        # (config_dir / "nnunet/results/nnUNet/2d").mkdir(exist_ok=True, parents=True)
        weights_dir = config_dir / "nnunet/results"

    # This variables will only be active during the python script execution. Therefore
    # we do not have to unset them in the end.
    os.environ["nnUNet_raw"] = str(weights_dir)  # not needed, just needs to be an existing directory
    os.environ["nnUNet_preprocessed"] = str(weights_dir)  # not needed, just needs to be an existing directory
    os.environ["nnUNet_results"] = str(weights_dir)


def setup_totalseg():
    totalseg_dir = get_totalseg_dir()
    totalseg_dir.mkdir(exist_ok=True)
    totalseg_config_file = totalseg_dir / "config.json"

    if totalseg_config_file.exists():
        with open(totalseg_config_file) as f:
            config = json.load(f)
    else:
        config = {
            "prediction_counter": 0
        }
        with open(totalseg_config_file, "w") as f:
            json.dump(config, f, indent=4)

    return config


def increase_prediction_counter():
    totalseg_dir = get_totalseg_dir()
    totalseg_config_file = totalseg_dir / "config.json"
    if totalseg_config_file.exists():
        with open(totalseg_config_file) as f:
            config = json.load(f)
        config["prediction_counter"] += 1
        with open(totalseg_config_file, "w") as f:
            json.dump(config, f, indent=4)
        return config


def get_config():
    totalseg_dir = get_totalseg_dir()
    totalseg_config_file = totalseg_dir / "config.json"
    if totalseg_config_file.exists():
        with open(totalseg_config_file) as f:
            config = json.load(f)
        return config
    else:
        return None


def get_version():
    try:
        return importlib.metadata.version("autoanatomy")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def get_config_key(key_name):
    totalseg_dir = get_totalseg_dir()
    totalseg_config_file = totalseg_dir / "config.json"
    if totalseg_config_file.exists():
        with open(totalseg_config_file) as f:
            config = json.load(f)
        if key_name in config:
            return config[key_name]
    return None


def set_config_key(key_name, value):
    totalseg_dir = get_totalseg_dir()
    totalseg_config_file = totalseg_dir / "config.json"
    if totalseg_config_file.exists():
        with open(totalseg_config_file) as f:
            config = json.load(f)
        config[key_name] = value
        with open(totalseg_config_file, "w") as f:
            json.dump(config, f, indent=4)
        return config
    else:
        print("WARNING: Could not set config key, because config file not found.")


