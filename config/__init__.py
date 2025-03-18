# 导入声优配置
import json
from importlib.resources import files


def load_command_config():
    config_path = files("config").joinpath("command_config.json")
    with open(config_path, "r") as f:
        return json.load(f)
