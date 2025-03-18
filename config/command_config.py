import json
from pathlib import Path
from config.voice_config import VoiceConfigManager


CONFIG_PATH = Path(__file__).parent / "command_config.json"


def load_dynamic_config():
    """加载动态配置"""
    try:
        # 加载基础配置
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            base_config = json.load(f)

        # 获取模型信息
        voice_config = VoiceConfigManager().get_models_info()

        # 更新 .load 命令的模型选项
        if ".load" in base_config["commands"]:
            base_config["commands"][".load"]["params"][0]["choices"] = list(
                voice_config["models"].keys()
            )

        # 动态设置每个模型的 voice 选项
        base_config["models"] = {}
        for model_repo in voice_config["models"]:
            base_config["models"][model_repo] = {
                "voices": [
                    v["name"] for v in voice_config["models"][model_repo]["voices"]
                ]
            }

        return base_config, voice_config
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as e:
        raise


class CommandConfig:
    def __init__(self):
        self.config, self.voice_config = load_dynamic_config()

        # 转换 .set 参数描述为字典（确保参数结构正确）
        set_params = self.config["commands"].get(".set", {}).get("params", [])
        if set_params:
            param_list = set_params[0].get("parameters", [])
            if isinstance(param_list, list):  # 如果参数是列表格式
                self.config["commands"][".set"]["params"][0]["parameters"] = {
                    param["name"]: param["description"] for param in param_list
                }

    def reload(self):
        """重新加载配置"""
        # logging.info("重新加载配置...")
        self.config, self.voice_config = load_dynamic_config()

    def get_config(self):
        return self.config

    def get_voice_options(self, model_repo: str) -> list:
        return self.voice_config["models"].get(model_repo, {}).get("voices", [])

    def get_param_values(self, param_name: str, model_repo: str = None) -> list:
        if param_name == "voice" and model_repo:
            return self.config["models"].get(model_repo, {}).get("voices", [])
        else:
            return self.config["param_values"].get(param_name, [])
